//! GPU-native, GPU-owned autoregressive token loop.
//!
//! Owns the execution of the entire forward transformer pass on GPU:
//! Embedding lookup -> Layer (Attention RMSNorm -> QKV/RoPE/KV -> Causal Attention/O -> MoE RMSNorm -> Router -> Q4 Expert Combine) -> Final RMSNorm -> LM Head -> GPU Greedy Argmax.

use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

use crate::architecture::Architecture;
use crate::backend::gpu_native::{
    GpuNativeAttentionGeometry, GpuNativeAttentionNorm, GpuNativeAttentionPlan,
    GpuNativeAttentionScratch, GpuNativeBootstrapError, GpuNativeDenseWeightHandle,
    GpuNativeDenseWeightKey, GpuNativeExecutorContext, GpuNativeKvState, GpuNativeQ4ExpertGeometry,
    GpuNativeQ4ExpertScratch, GpuNativeRmsNormHandle, GpuNativeRouterGeometry, GpuNativeRouterPlan,
    GpuNativeRouterScratch, GpuNativeScratch, GpuNativeTokenState, GPU_NATIVE_STATUS_FATAL_MASK,
    GPU_NATIVE_STATUS_RETRYABLE_MASK, MAX_GPU_NATIVE_ROUTER_EXPERTS, MAX_GPU_NATIVE_ROUTER_TOP_K,
};
use crate::dense_tensor::DenseDType;
use crate::engine::{Engine, GpuNativeDemandResidencyError};
use crate::gating::ScoringFunc;
use crate::gpu_native_residency::GpuNativeTieredResidencyManager;
use crate::model::RealModel;
use crate::sampling::SamplingParams;

/// Structured summary of runtime counters across the GPU-native token loop.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct GpuNativeTokenLoopSnapshot {
    pub token_attempts: u64,
    pub tokens_completed: u64,
    pub warm_tokens_completed: u64,
    pub residency_miss_attempts: u64,
    pub replay_attempts: u64,
    pub residency_services: u64,
    pub fatal_failures: u64,
    pub no_progress_failures: u64,
    pub queue_submissions: u64,
    pub boundary_maps: u64,
    pub boundary_readbacks: u64,
}

#[derive(Default)]
struct GpuNativeTokenLoopCounters {
    token_attempts: AtomicU64,
    tokens_completed: AtomicU64,
    warm_tokens_completed: AtomicU64,
    residency_miss_attempts: AtomicU64,
    replay_attempts: AtomicU64,
    residency_services: AtomicU64,
    fatal_failures: AtomicU64,
    no_progress_failures: AtomicU64,
    queue_submissions: AtomicU64,
    boundary_maps: AtomicU64,
    boundary_readbacks: AtomicU64,
}

impl GpuNativeTokenLoopCounters {
    fn snapshot(&self) -> GpuNativeTokenLoopSnapshot {
        GpuNativeTokenLoopSnapshot {
            token_attempts: self.token_attempts.load(Ordering::Relaxed),
            tokens_completed: self.tokens_completed.load(Ordering::Relaxed),
            warm_tokens_completed: self.warm_tokens_completed.load(Ordering::Relaxed),
            residency_miss_attempts: self.residency_miss_attempts.load(Ordering::Relaxed),
            replay_attempts: self.replay_attempts.load(Ordering::Relaxed),
            residency_services: self.residency_services.load(Ordering::Relaxed),
            fatal_failures: self.fatal_failures.load(Ordering::Relaxed),
            no_progress_failures: self.no_progress_failures.load(Ordering::Relaxed),
            queue_submissions: self.queue_submissions.load(Ordering::Relaxed),
            boundary_maps: self.boundary_maps.load(Ordering::Relaxed),
            boundary_readbacks: self.boundary_readbacks.load(Ordering::Relaxed),
        }
    }
}

/// Errors originating in the GPU-native token loop.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuNativeTokenLoopError {
    IncompatibleModel(GpuNativeModelCompatibilityError),
    ContextLimitExceeded {
        requested_position: usize,
        max_seq_len: usize,
    },
    AttemptBoundExceeded {
        attempts: usize,
        max_attempts: usize,
    },
    NoProgress {
        layer_index: usize,
        selected_ids: Vec<u32>,
    },
    FatalNumericalFailure {
        layer_index: Option<usize>,
        status_bits: u32,
    },
    ResidencyServiceFailed(GpuNativeDemandResidencyError),
    UnsupportedSampling {
        reason: String,
    },
    Bootstrap(GpuNativeBootstrapError),
    InvalidBoundaryReport {
        detail: String,
    },
    InvalidSelectedExpertId {
        layer_index: usize,
        expert_id: u32,
    },
    DuplicateSelectedExpertId {
        layer_index: usize,
        expert_id: u32,
    },
    InvalidTopKCount {
        expected: usize,
        actual: usize,
    },
    MapFailed(String),
}

impl fmt::Display for GpuNativeTokenLoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleModel(err) => write!(f, "incompatible model: {err}"),
            Self::ContextLimitExceeded {
                requested_position,
                max_seq_len,
            } => write!(
                f,
                "requested position {requested_position} exceeds gpu_native_max_seq_len {max_seq_len}"
            ),
            Self::AttemptBoundExceeded {
                attempts,
                max_attempts,
            } => write!(
                f,
                "token attempt bound exceeded: {attempts} attempts >= {max_attempts}"
            ),
            Self::NoProgress {
                layer_index,
                selected_ids,
            } => write!(
                f,
                "no progress after residency service on layer {layer_index} with experts {selected_ids:?}"
            ),
            Self::FatalNumericalFailure {
                layer_index,
                status_bits,
            } => write!(
                f,
                "fatal numerical failure on {:?} with status bits 0x{status_bits:08x}",
                layer_index
            ),
            Self::ResidencyServiceFailed(err) => write!(f, "residency service failed: {err}"),
            Self::UnsupportedSampling { reason } => write!(f, "unsupported sampling: {reason}"),
            Self::Bootstrap(err) => write!(f, "GPU-native bootstrap error: {err}"),
            Self::InvalidBoundaryReport { detail } => {
                write!(f, "invalid boundary report: {detail}")
            }
            Self::InvalidSelectedExpertId {
                layer_index,
                expert_id,
            } => write!(
                f,
                "invalid selected expert id {expert_id} on layer {layer_index}"
            ),
            Self::DuplicateSelectedExpertId {
                layer_index,
                expert_id,
            } => write!(
                f,
                "duplicate selected expert id {expert_id} on layer {layer_index}"
            ),
            Self::InvalidTopKCount { expected, actual } => write!(
                f,
                "selected expert count mismatch: expected {expected}, got {actual}"
            ),
            Self::MapFailed(detail) => write!(f, "staging buffer map failed: {detail}"),
        }
    }
}

impl std::error::Error for GpuNativeTokenLoopError {}

impl From<GpuNativeModelCompatibilityError> for GpuNativeTokenLoopError {
    fn from(err: GpuNativeModelCompatibilityError) -> Self {
        Self::IncompatibleModel(err)
    }
}

impl From<GpuNativeBootstrapError> for GpuNativeTokenLoopError {
    fn from(err: GpuNativeBootstrapError) -> Self {
        Self::Bootstrap(err)
    }
}

/// Errors raised when a model checkpoint violates the supported GPU-native model contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuNativeModelCompatibilityError {
    UnsupportedArchitecture {
        architecture: String,
    },
    DenseLayerUnsupported {
        layer_index: usize,
    },
    SharedExpertUnsupported {
        layer_index: usize,
    },
    MlaUnsupported {
        layer_index: usize,
    },
    NonQ4ExpertDtype {
        dtype: String,
    },
    TooManyExperts {
        num_experts: usize,
        max: usize,
    },
    InvalidTopK {
        top_k: usize,
        max: usize,
    },
    IncompatibleGeometry {
        detail: String,
    },
    AsymmetricVHeadDim {
        head_dim: usize,
        v_head_dim: usize,
    },
    AttentionSinkUnsupported {
        layer_index: usize,
    },
    AttentionBiasesUnsupported {
        layer_index: usize,
    },
    ValueScaleUnsupported {
        layer_index: usize,
    },
    SlidingWindowUnsupported {
        layer_index: usize,
    },
    GroupedRoutingUnsupported {
        layer_index: usize,
    },
    NonSoftmaxRouter {
        layer_index: usize,
    },
    RouterCorrectionBiasUnsupported {
        layer_index: usize,
    },
    RoutedScalingFactorUnsupported {
        layer_index: usize,
        factor_bits: u32,
    },
    NonNormalisedTopK {
        layer_index: usize,
    },
    UnsupportedDenseDtype {
        tensor: String,
        dtype: String,
    },
}

impl fmt::Display for GpuNativeModelCompatibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedArchitecture { architecture } => write!(
                f,
                "GPU-native execution supports only Qwen3Moe in this slice, got {architecture}"
            ),
            Self::DenseLayerUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} contains a dense FFN; all layers must be sparse MoE"
            ),
            Self::SharedExpertUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} contains a shared expert; shared experts are unsupported"
            ),
            Self::MlaUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} contains MLA; MLA is unsupported"
            ),
            Self::NonQ4ExpertDtype { dtype } => write!(
                f,
                "routed expert dtype must be Q4_0, got {dtype}"
            ),
            Self::TooManyExperts { num_experts, max } => write!(
                f,
                "num_experts ({num_experts}) exceeds GPU router maximum ({max})"
            ),
            Self::InvalidTopK { top_k, max } => write!(
                f,
                "top_k ({top_k}) must be in 1..={max}"
            ),
            Self::IncompatibleGeometry { detail } => write!(
                f,
                "incompatible model geometry: {detail}"
            ),
            Self::AsymmetricVHeadDim { head_dim, v_head_dim } => write!(
                f,
                "asymmetric V head dimension unsupported: head_dim={head_dim}, v_head_dim={v_head_dim}"
            ),
            Self::AttentionSinkUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} uses attention sink bias; sinks are unsupported"
            ),
            Self::AttentionBiasesUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} uses attention projection biases; biases are unsupported"
            ),
            Self::ValueScaleUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} uses attention value scaling; value scaling is unsupported"
            ),
            Self::SlidingWindowUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} uses sliding-window attention; sliding window is unsupported"
            ),
            Self::GroupedRoutingUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} uses grouped expert routing; grouped routing is unsupported"
            ),
            Self::NonSoftmaxRouter { layer_index } => write!(
                f,
                "layer {layer_index} uses non-softmax router scoring; only Softmax is supported"
            ),
            Self::RouterCorrectionBiasUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} uses router correction bias; correction bias is unsupported"
            ),
            Self::RoutedScalingFactorUnsupported { layer_index, factor_bits } => write!(
                f,
                "layer {layer_index} uses routed scaling factor 0x{factor_bits:08x} != 1.0"
            ),
            Self::NonNormalisedTopK { layer_index } => write!(
                f,
                "layer {layer_index} does not normalise top-K weights; top-K normalisation is required"
            ),
            Self::UnsupportedDenseDtype { tensor, dtype } => write!(
                f,
                "tensor {tensor} has unsupported dense dtype {dtype}; only F32 and Q8_0 are supported"
            ),
        }
    }
}

impl std::error::Error for GpuNativeModelCompatibilityError {}

/// Fixed byte layout for one token-boundary compact report staging buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuNativeBoundaryReportLayout {
    pub num_layers: usize,
    pub top_k: usize,
    pub layer_status_offset: usize,
    pub layer_status_bytes: usize,
    pub selected_ids_offset: usize,
    pub selected_ids_bytes: usize,
    pub final_status_offset: usize,
    pub sampled_token_offset: usize,
    pub total_bytes: u64,
}

impl GpuNativeBoundaryReportLayout {
    pub fn try_new(num_layers: usize, top_k: usize) -> Result<Self, GpuNativeTokenLoopError> {
        if num_layers == 0 {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "num_layers must be > 0".into(),
            });
        }
        if top_k == 0 || top_k > MAX_GPU_NATIVE_ROUTER_TOP_K {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: format!("top_k must be in 1..={MAX_GPU_NATIVE_ROUTER_TOP_K}"),
            });
        }
        let u32_bytes = std::mem::size_of::<u32>();

        let layer_status_offset = 0usize;
        let layer_status_bytes = num_layers.checked_mul(u32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "layer status bytes overflow".into(),
            }
        })?;

        let selected_ids_offset = layer_status_bytes;
        let selected_ids_per_layer = top_k.checked_mul(u32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "selected ids per layer overflow".into(),
            }
        })?;
        let selected_ids_bytes =
            num_layers
                .checked_mul(selected_ids_per_layer)
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "total selected ids bytes overflow".into(),
                })?;

        let final_status_offset = selected_ids_offset
            .checked_add(selected_ids_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "final status offset overflow".into(),
            })?;

        let sampled_token_offset = final_status_offset.checked_add(u32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "sampled token offset overflow".into(),
            }
        })?;

        let total_size = sampled_token_offset.checked_add(u32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "total report bytes overflow".into(),
            }
        })?;

        let total_bytes = u64::try_from(total_size).map_err(|_| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "total report bytes exceed u64".into(),
            }
        })?;

        Ok(Self {
            num_layers,
            top_k,
            layer_status_offset,
            layer_status_bytes,
            selected_ids_offset,
            selected_ids_bytes,
            final_status_offset,
            sampled_token_offset,
            total_bytes,
        })
    }

    pub fn parse(&self, bytes: &[u8]) -> Result<GpuNativeBoundaryReport, GpuNativeTokenLoopError> {
        if (bytes.len() as u64) < self.total_bytes {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: format!(
                    "readback bytes length {} less than required {}",
                    bytes.len(),
                    self.total_bytes
                ),
            });
        }

        let mut layer_statuses = Vec::with_capacity(self.num_layers);
        for l in 0..self.num_layers {
            let start = self.layer_status_offset + l * 4;
            let status = u32::from_le_bytes(bytes[start..start + 4].try_into().map_err(|_| {
                GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "failed to decode layer status".into(),
                }
            })?);
            layer_statuses.push(status);
        }

        let mut selected_ids = Vec::with_capacity(self.num_layers);
        for l in 0..self.num_layers {
            let mut layer_ids = Vec::with_capacity(self.top_k);
            let layer_start = self.selected_ids_offset + l * self.top_k * 4;
            for k in 0..self.top_k {
                let start = layer_start + k * 4;
                let id = u32::from_le_bytes(bytes[start..start + 4].try_into().map_err(|_| {
                    GpuNativeTokenLoopError::InvalidBoundaryReport {
                        detail: "failed to decode selected id".into(),
                    }
                })?);
                layer_ids.push(id);
            }
            selected_ids.push(layer_ids);
        }

        let final_status = u32::from_le_bytes(
            bytes[self.final_status_offset..self.final_status_offset + 4]
                .try_into()
                .map_err(|_| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "failed to decode final status".into(),
                })?,
        );

        let sampled_token = u32::from_le_bytes(
            bytes[self.sampled_token_offset..self.sampled_token_offset + 4]
                .try_into()
                .map_err(|_| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "failed to decode sampled token".into(),
                })?,
        );

        Ok(GpuNativeBoundaryReport {
            layer_statuses,
            selected_ids,
            final_status,
            sampled_token,
        })
    }
}

/// Parsed results of one token attempt's boundary readback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuNativeBoundaryReport {
    pub layer_statuses: Vec<u32>,
    pub selected_ids: Vec<Vec<u32>>,
    pub final_status: u32,
    pub sampled_token: u32,
}

impl GpuNativeBoundaryReport {
    /// Returns the index of the first layer whose latched status transitioned to non-zero.
    pub fn first_failure_layer(&self) -> Option<usize> {
        self.layer_statuses.iter().position(|&status| status != 0)
    }
}

/// Persistent per-layer plans and handles for one Qwen3-MoE transformer layer.
pub struct GpuNativeLayerPlan {
    pub layer_index: usize,
    pub rms_attn_handle: GpuNativeRmsNormHandle,
    pub rms_moe_handle: GpuNativeRmsNormHandle,
    pub attn_plan: GpuNativeAttentionPlan,
    pub router_plan: GpuNativeRouterPlan,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuNativeModelGeometry {
    pub num_layers: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
}

/// Persistent, model-scoped owner of the GPU-native token loop.
pub struct GpuNativeTokenLoop {
    executor: Arc<GpuNativeExecutorContext>,
    residency_manager: Arc<GpuNativeTieredResidencyManager>,
    model_geometry: GpuNativeModelGeometry,
    embedding_handle: GpuNativeDenseWeightHandle,
    final_norm_handle: GpuNativeRmsNormHandle,
    lm_head_handle: GpuNativeDenseWeightHandle,
    layers: Vec<GpuNativeLayerPlan>,
    report_layout: GpuNativeBoundaryReportLayout,
    counters: GpuNativeTokenLoopCounters,
    execution_guard: TokioMutex<()>,
}

impl GpuNativeTokenLoop {
    /// Validate that `model` conforms strictly to the supported Qwen3Moe GPU-native contract.
    pub fn validate_model_compatibility(
        model: &RealModel,
    ) -> Result<(), GpuNativeModelCompatibilityError> {
        if model.config.architecture != Architecture::Qwen3Moe {
            return Err(GpuNativeModelCompatibilityError::UnsupportedArchitecture {
                architecture: format!("{:?}", model.config.architecture),
            });
        }
        if model.config.top_k == 0 || model.config.top_k > MAX_GPU_NATIVE_ROUTER_TOP_K {
            return Err(GpuNativeModelCompatibilityError::InvalidTopK {
                top_k: model.config.top_k,
                max: MAX_GPU_NATIVE_ROUTER_TOP_K,
            });
        }
        if model.config.num_experts > MAX_GPU_NATIVE_ROUTER_EXPERTS {
            return Err(GpuNativeModelCompatibilityError::TooManyExperts {
                num_experts: model.config.num_experts,
                max: MAX_GPU_NATIVE_ROUTER_EXPERTS,
            });
        }
        if model.config.window_size.is_some() {
            return Err(GpuNativeModelCompatibilityError::SlidingWindowUnsupported {
                layer_index: 0,
            });
        }

        let is_valid_dense_dtype =
            |dtype: DenseDType| -> bool { matches!(dtype, DenseDType::F32 | DenseDType::Q8_0) };

        if !is_valid_dense_dtype(model.embedding.dtype()) {
            return Err(GpuNativeModelCompatibilityError::UnsupportedDenseDtype {
                tensor: "embed.weight".into(),
                dtype: model.embedding.dtype_name().into(),
            });
        }
        if !is_valid_dense_dtype(model.lm_head.weights.dtype()) {
            return Err(GpuNativeModelCompatibilityError::UnsupportedDenseDtype {
                tensor: "lm_head.weight".into(),
                dtype: model.lm_head.weights.dtype_name().into(),
            });
        }
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            if layer.dense_ffn.is_some() {
                return Err(GpuNativeModelCompatibilityError::DenseLayerUnsupported {
                    layer_index: layer_idx,
                });
            }
            if layer.shared_expert.is_some() {
                return Err(GpuNativeModelCompatibilityError::SharedExpertUnsupported {
                    layer_index: layer_idx,
                });
            }
            if layer.mla.is_some() {
                return Err(GpuNativeModelCompatibilityError::MlaUnsupported {
                    layer_index: layer_idx,
                });
            }
            if layer.attn.v_head_dim != layer.attn.head_dim {
                return Err(GpuNativeModelCompatibilityError::AsymmetricVHeadDim {
                    head_dim: layer.attn.head_dim,
                    v_head_dim: layer.attn.v_head_dim,
                });
            }
            if layer.attn.sink_bias.is_some() {
                return Err(GpuNativeModelCompatibilityError::AttentionSinkUnsupported {
                    layer_index: layer_idx,
                });
            }
            if layer.attn.bq.is_some()
                || layer.attn.bk.is_some()
                || layer.attn.bv.is_some()
                || layer.attn.bo.is_some()
            {
                return Err(
                    GpuNativeModelCompatibilityError::AttentionBiasesUnsupported {
                        layer_index: layer_idx,
                    },
                );
            }
            if layer.attn.attention_value_scale.is_some() {
                return Err(GpuNativeModelCompatibilityError::ValueScaleUnsupported {
                    layer_index: layer_idx,
                });
            }
            if layer.attn.window_size.is_some() {
                return Err(GpuNativeModelCompatibilityError::SlidingWindowUnsupported {
                    layer_index: layer_idx,
                });
            }
            if layer.gate.scoring_func != ScoringFunc::Softmax {
                return Err(GpuNativeModelCompatibilityError::NonSoftmaxRouter {
                    layer_index: layer_idx,
                });
            }
            if layer.gate.correction_bias.is_some() {
                return Err(
                    GpuNativeModelCompatibilityError::RouterCorrectionBiasUnsupported {
                        layer_index: layer_idx,
                    },
                );
            }
            if layer.gate.n_group > 1 || layer.gate.topk_group > 1 {
                return Err(
                    GpuNativeModelCompatibilityError::GroupedRoutingUnsupported {
                        layer_index: layer_idx,
                    },
                );
            }
            if (layer.gate.routed_scaling_factor - 1.0).abs() > 1e-5 {
                return Err(
                    GpuNativeModelCompatibilityError::RoutedScalingFactorUnsupported {
                        layer_index: layer_idx,
                        factor_bits: layer.gate.routed_scaling_factor.to_bits(),
                    },
                );
            }
            if !layer.gate.normalise_topk {
                return Err(GpuNativeModelCompatibilityError::NonNormalisedTopK {
                    layer_index: layer_idx,
                });
            }

            for (name, weight) in [
                ("wq", &layer.attn.wq),
                ("wk", &layer.attn.wk),
                ("wv", &layer.attn.wv),
                ("wo", &layer.attn.wo),
                ("gate", &layer.gate.weights),
            ] {
                if !is_valid_dense_dtype(weight.dtype()) {
                    return Err(GpuNativeModelCompatibilityError::UnsupportedDenseDtype {
                        tensor: format!("layer_{layer_idx}_{name}"),
                        dtype: weight.dtype_name().into(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Construct and initialize the persistent GPU-native token loop.
    pub fn try_new(
        executor: Arc<GpuNativeExecutorContext>,
        residency_manager: Arc<GpuNativeTieredResidencyManager>,
        model: &RealModel,
        max_seq_len: usize,
    ) -> Result<Arc<Self>, GpuNativeTokenLoopError> {
        Self::validate_model_compatibility(model)?;

        let num_layers = model.layers.len();
        let top_k = model.config.top_k;
        let d_model = model.config.d_model;
        let d_ff = model.config.d_ff;
        let num_experts = model.config.num_experts;
        let num_heads = model.config.num_heads;
        let num_kv_heads = model.config.num_kv_heads;
        let head_dim = model.config.head_dim;
        let vocab_size = model.config.vocab_size;
        let rms_eps = model.config.rms_eps;
        let rope_base = model.config.rope_base;

        if max_seq_len == 0 {
            return Err(GpuNativeTokenLoopError::ContextLimitExceeded {
                requested_position: 0,
                max_seq_len,
            });
        }

        let model_geometry = GpuNativeModelGeometry {
            num_layers,
            d_model,
            d_ff,
            num_experts,
            top_k,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            max_seq_len,
            rms_eps,
            rope_base,
        };

        let report_layout = GpuNativeBoundaryReportLayout::try_new(num_layers, top_k)?;

        let embedding_handle = executor.register_dense_weight(
            GpuNativeDenseWeightKey::try_new("model.embed")?,
            &model.embedding,
        )?;

        let mut layers = Vec::with_capacity(num_layers);
        let router_geometry = GpuNativeRouterGeometry::try_new(d_model, num_experts, top_k)?;
        let attention_geometry = GpuNativeAttentionGeometry::try_new(
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
        )?;

        for (l, layer) in model.layers.iter().enumerate() {
            let rms_attn_handle = executor.register_rms_norm(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.rms_attn"))?,
                layer.rms_attn.weight.as_slice(),
            )?;
            let rms_moe_handle = executor.register_rms_norm(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.rms_moe"))?,
                layer.rms_moe.weight.as_slice(),
            )?;

            let q_handle = executor.register_dense_weight(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.attn.q"))?,
                &layer.attn.wq,
            )?;
            let k_handle = executor.register_dense_weight(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.attn.k"))?,
                &layer.attn.wk,
            )?;
            let v_handle = executor.register_dense_weight(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.attn.v"))?,
                &layer.attn.wv,
            )?;
            let o_handle = executor.register_dense_weight(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.attn.o"))?,
                &layer.attn.wo,
            )?;

            let q_norm_handle = if let Some(ref q_norm) = layer.attn.q_norm {
                let handle = executor.register_rms_norm(
                    GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.attn.q_norm"))?,
                    q_norm.weight.as_slice(),
                )?;
                Some(GpuNativeAttentionNorm::try_new(handle, q_norm.eps)?)
            } else {
                None
            };

            let k_norm_handle = if let Some(ref k_norm) = layer.attn.k_norm {
                let handle = executor.register_rms_norm(
                    GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.attn.k_norm"))?,
                    k_norm.weight.as_slice(),
                )?;
                Some(GpuNativeAttentionNorm::try_new(handle, k_norm.eps)?)
            } else {
                None
            };

            let rope_handle = executor.register_standard_rope(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.attn.rope"))?,
                layer.attn.rope_dim,
                layer.attn.rope_base,
            )?;

            let attn_plan = executor.create_attention_plan(
                l,
                attention_geometry,
                q_handle,
                k_handle,
                v_handle,
                o_handle,
                q_norm_handle,
                k_norm_handle,
                rope_handle,
            )?;

            let gate_handle = executor.register_dense_weight(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.router"))?,
                &layer.gate.weights,
            )?;
            let router_plan = executor.create_router_plan(l, router_geometry, gate_handle)?;

            layers.push(GpuNativeLayerPlan {
                layer_index: l,
                rms_attn_handle,
                rms_moe_handle,
                attn_plan,
                router_plan,
            });
        }

        let final_norm_handle = executor.register_rms_norm(
            GpuNativeDenseWeightKey::try_new("model.final_norm")?,
            model.final_rms.weight.as_slice(),
        )?;
        let lm_head_handle = executor.register_dense_weight(
            GpuNativeDenseWeightKey::try_new("model.lm_head")?,
            &model.lm_head.weights,
        )?;

        Ok(Arc::new(Self {
            executor,
            residency_manager,
            model_geometry,
            embedding_handle,
            final_norm_handle,
            lm_head_handle,
            layers,
            report_layout,
            counters: GpuNativeTokenLoopCounters::default(),
            execution_guard: TokioMutex::new(()),
        }))
    }

    pub fn model_geometry(&self) -> GpuNativeModelGeometry {
        self.model_geometry
    }

    pub fn max_seq_len(&self) -> usize {
        self.model_geometry.max_seq_len
    }

    pub fn snapshot(&self) -> GpuNativeTokenLoopSnapshot {
        self.counters.snapshot()
    }

    /// Allocate request-local device resources for one GPU-native sequence.
    pub fn create_request_state(&self) -> Result<GpuNativeRequestState, GpuNativeTokenLoopError> {
        let token_state = self.executor.create_token_state()?;
        let kv_width = self.model_geometry.num_kv_heads * self.model_geometry.head_dim;
        let kv_state = self.executor.create_kv_state(
            self.model_geometry.num_layers,
            self.model_geometry.max_seq_len,
            kv_width,
        )?;

        let attn_geom = GpuNativeAttentionGeometry::try_new(
            self.model_geometry.d_model,
            self.model_geometry.num_heads,
            self.model_geometry.num_kv_heads,
            self.model_geometry.head_dim,
            self.model_geometry.max_seq_len,
        )?;
        let attn_scratch = self.executor.create_attention_scratch(attn_geom)?;

        let router_geom = GpuNativeRouterGeometry::try_new(
            self.model_geometry.d_model,
            self.model_geometry.num_experts,
            self.model_geometry.top_k,
        )?;
        let router_scratch = self.executor.create_router_scratch(router_geom)?;

        let expert_geom = GpuNativeQ4ExpertGeometry::try_new(
            self.model_geometry.d_model,
            self.model_geometry.d_ff,
            self.model_geometry.num_experts,
            self.model_geometry.top_k,
        )?;
        let expert_scratch = self.executor.create_q4_expert_scratch(expert_geom)?;

        let logits_scratch = self
            .executor
            .create_scratch(self.model_geometry.vocab_size)?;
        let sampled_token_buf = self.executor.create_scratch(1)?;

        let gpu = self.executor.authoritative_gpu()?;
        let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_boundary_staging"),
            size: self.report_layout.total_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuNativeRequestState {
            token_state,
            kv_state,
            attn_scratch,
            router_scratch,
            expert_scratch,
            logits_scratch,
            sampled_token_buf,
            staging_buffer,
            committed_position: 0,
            max_seq_len: self.model_geometry.max_seq_len,
        })
    }

    /// Ingest a multi-token prompt, commit KV positions, and generate up to `max_tokens` completion tokens.
    pub async fn ingest_prompt_and_generate(
        self: &Arc<Self>,
        engine: &Arc<Engine>,
        request: &mut GpuNativeRequestState,
        prompt_ids: &[u32],
        max_tokens: usize,
        params: &SamplingParams,
    ) -> Result<Vec<u32>, GpuNativeTokenLoopError> {
        if prompt_ids.is_empty() {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "prompt_ids must be non-empty".into(),
            });
        }
        if !params.is_greedy() {
            return Err(GpuNativeTokenLoopError::UnsupportedSampling {
                reason: "only greedy sampling (temperature=0.0) is supported in gpu_native mode in this slice".into(),
            });
        }
        let total_required_len = prompt_ids.len().checked_add(max_tokens).ok_or_else(|| {
            GpuNativeTokenLoopError::ContextLimitExceeded {
                requested_position: usize::MAX,
                max_seq_len: request.max_seq_len,
            }
        })?;
        if total_required_len > request.max_seq_len {
            return Err(GpuNativeTokenLoopError::ContextLimitExceeded {
                requested_position: total_required_len,
                max_seq_len: request.max_seq_len,
            });
        }

        let _guard = self.execution_guard.lock().await;

        // Ingest prefix prompt tokens without evaluating LM-head
        let prefix_count = prompt_ids.len().saturating_sub(1);
        for &token_id in &prompt_ids[..prefix_count] {
            let pos = request.committed_position;
            self.step_token(engine, request, token_id, pos, false)
                .await?;
        }

        // Final prompt token: evaluate LM-head and sample first completion token
        let final_prompt = *prompt_ids.last().expect("checked non-empty");
        let final_prompt_pos = request.committed_position;
        let first_completion = self
            .step_token(engine, request, final_prompt, final_prompt_pos, true)
            .await?
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "final prompt step produced no sampled token".into(),
            })?;

        let mut completion_ids = Vec::with_capacity(max_tokens);
        completion_ids.push(first_completion);

        let mut last_token = first_completion;
        while completion_ids.len() < max_tokens {
            let pos = request.committed_position;
            if pos >= request.max_seq_len {
                break;
            }
            let next_token = self
                .step_token(engine, request, last_token, pos, true)
                .await?
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "decode step produced no sampled token".into(),
                })?;
            completion_ids.push(next_token);
            last_token = next_token;
        }

        Ok(completion_ids)
    }

    /// Execute one single token step with bounded retries and residency demand service.
    pub async fn step_token(
        &self,
        engine: &Arc<Engine>,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        sample: bool,
    ) -> Result<Option<u32>, GpuNativeTokenLoopError> {
        let mut attempt = 0;
        let mut last_miss_sig: Option<(usize, Vec<u32>)> = None;
        let max_attempts = self.layers.len() + 1;
        let mut is_warm = true;

        loop {
            if attempt >= max_attempts {
                return Err(GpuNativeTokenLoopError::AttemptBoundExceeded {
                    attempts: attempt,
                    max_attempts,
                });
            }

            let replay = attempt > 0;
            let report = self.execute_token_attempt(request, token_id, position, sample, replay)?;

            if let Some(fail_layer) = report.first_failure_layer() {
                let layer_status = report.layer_statuses[fail_layer];
                if (layer_status & GPU_NATIVE_STATUS_FATAL_MASK) != 0 {
                    self.counters.fatal_failures.fetch_add(1, Ordering::Relaxed);
                    return Err(GpuNativeTokenLoopError::FatalNumericalFailure {
                        layer_index: Some(fail_layer),
                        status_bits: layer_status,
                    });
                }
                if (layer_status & GPU_NATIVE_STATUS_RETRYABLE_MASK)
                    == GPU_NATIVE_STATUS_RETRYABLE_MASK
                {
                    is_warm = false;
                    self.counters
                        .residency_miss_attempts
                        .fetch_add(1, Ordering::Relaxed);
                    let local_ids = &report.selected_ids[fail_layer];

                    for &id in local_ids {
                        if id as usize >= self.model_geometry.num_experts {
                            return Err(GpuNativeTokenLoopError::InvalidSelectedExpertId {
                                layer_index: fail_layer,
                                expert_id: id,
                            });
                        }
                    }
                    let mut seen = HashSet::with_capacity(local_ids.len());
                    for &id in local_ids {
                        if !seen.insert(id) {
                            return Err(GpuNativeTokenLoopError::DuplicateSelectedExpertId {
                                layer_index: fail_layer,
                                expert_id: id,
                            });
                        }
                    }
                    if local_ids.len() != self.model_geometry.top_k {
                        return Err(GpuNativeTokenLoopError::InvalidTopKCount {
                            expected: self.model_geometry.top_k,
                            actual: local_ids.len(),
                        });
                    }

                    if let Some((prev_layer, ref prev_ids)) = last_miss_sig {
                        if prev_layer == fail_layer && prev_ids == local_ids {
                            self.counters
                                .no_progress_failures
                                .fetch_add(1, Ordering::Relaxed);
                            return Err(GpuNativeTokenLoopError::NoProgress {
                                layer_index: fail_layer,
                                selected_ids: local_ids.clone(),
                            });
                        }
                    }

                    let global_ids: Vec<u32> = local_ids
                        .iter()
                        .map(|&loc| {
                            fail_layer as u32 * self.model_geometry.num_experts as u32 + loc
                        })
                        .collect();

                    engine
                        .ensure_gpu_native_demand_residency(fail_layer, &global_ids)
                        .await
                        .map_err(GpuNativeTokenLoopError::ResidencyServiceFailed)?;

                    self.counters
                        .residency_services
                        .fetch_add(1, Ordering::Relaxed);
                    last_miss_sig = Some((fail_layer, local_ids.clone()));
                    attempt += 1;
                    continue;
                }

                return Err(GpuNativeTokenLoopError::FatalNumericalFailure {
                    layer_index: Some(fail_layer),
                    status_bits: layer_status,
                });
            }

            if (report.final_status & GPU_NATIVE_STATUS_FATAL_MASK) != 0 {
                self.counters.fatal_failures.fetch_add(1, Ordering::Relaxed);
                return Err(GpuNativeTokenLoopError::FatalNumericalFailure {
                    layer_index: None,
                    status_bits: report.final_status,
                });
            }
            if report.final_status != 0 {
                return Err(GpuNativeTokenLoopError::FatalNumericalFailure {
                    layer_index: None,
                    status_bits: report.final_status,
                });
            }

            request.committed_position += 1;
            self.counters
                .tokens_completed
                .fetch_add(1, Ordering::Relaxed);
            if is_warm {
                self.counters
                    .warm_tokens_completed
                    .fetch_add(1, Ordering::Relaxed);
            }

            engine.record_gpu_native_actual_routes(&report.selected_ids);

            if sample {
                return Ok(Some(report.sampled_token));
            } else {
                return Ok(None);
            }
        }
    }

    /// Encode and execute one single attempt on device and read back the compact boundary report.
    pub fn execute_token_attempt(
        &self,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        sample: bool,
        replay: bool,
    ) -> Result<GpuNativeBoundaryReport, GpuNativeTokenLoopError> {
        if position != request.committed_position {
            return Err(GpuNativeTokenLoopError::ContextLimitExceeded {
                requested_position: position,
                max_seq_len: request.committed_position,
            });
        }
        if position >= request.max_seq_len {
            return Err(GpuNativeTokenLoopError::ContextLimitExceeded {
                requested_position: position,
                max_seq_len: request.max_seq_len,
            });
        }

        self.counters.token_attempts.fetch_add(1, Ordering::Relaxed);
        self.counters
            .queue_submissions
            .fetch_add(1, Ordering::Relaxed);
        self.counters.boundary_maps.fetch_add(1, Ordering::Relaxed);
        self.counters
            .boundary_readbacks
            .fetch_add(1, Ordering::Relaxed);
        if replay {
            self.counters
                .replay_attempts
                .fetch_add(1, Ordering::Relaxed);
        }

        let gpu = self.executor.authoritative_gpu()?;
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_native_token_attempt"),
            });

        if replay {
            self.executor
                .encode_clear_retryable_status(&mut encoder, &request.token_state)?;
        }

        self.executor.encode_embedding_lookup(
            &mut encoder,
            &self.embedding_handle,
            token_id,
            &request.token_state,
        )?;

        let num_layers = self.layers.len();
        let top_k = self.model_geometry.top_k;
        let rms_eps = self.model_geometry.rms_eps;

        for (layer_idx, layer_plan) in self.layers.iter().enumerate() {
            // 1. Attention Pre-Norm
            self.executor.encode_rms_norm_state_in_place(
                &mut encoder,
                &layer_plan.rms_attn_handle,
                rms_eps,
                &request.token_state,
            )?;

            // 2. Attention Prepare
            self.executor.encode_attention_prepare(
                &mut encoder,
                &layer_plan.attn_plan,
                &request.token_state,
                &request.attn_scratch,
                &request.kv_state,
                position,
            )?;

            // 3. Attention Complete
            self.executor.encode_attention_complete(
                &mut encoder,
                &layer_plan.attn_plan,
                &request.token_state,
                &request.attn_scratch,
                &request.kv_state,
                position + 1,
            )?;

            // 4. MoE Pre-Norm
            self.executor.encode_rms_norm_state_in_place(
                &mut encoder,
                &layer_plan.rms_moe_handle,
                rms_eps,
                &request.token_state,
            )?;

            // 5. Router
            self.executor.encode_router(
                &mut encoder,
                &layer_plan.router_plan,
                &request.token_state,
                &request.router_scratch,
            )?;

            // 6. Expert Combine
            let arena = self.residency_manager.arena(layer_idx).ok_or_else(|| {
                GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: format!("missing expert arena for layer {layer_idx}"),
                }
            })?;
            self.executor.encode_q4_expert_arena_combine(
                &mut encoder,
                &layer_plan.router_plan,
                &request.router_scratch,
                arena,
                &request.token_state,
                &request.expert_scratch,
            )?;

            // Copy layer status to staging
            let status_offset = (layer_idx * 4) as u64;
            encoder.copy_buffer_to_buffer(
                request.token_state.status_buffer(),
                0,
                &request.staging_buffer,
                status_offset,
                4,
            );

            // Copy selected ids to staging before scratch reuse
            let ids_offset = (num_layers * 4 + layer_idx * top_k * 4) as u64;
            encoder.copy_buffer_to_buffer(
                request.router_scratch.selected_ids_buffer(),
                0,
                &request.staging_buffer,
                ids_offset,
                (top_k * 4) as u64,
            );
        }

        if sample {
            // Final RMSNorm
            self.executor.encode_rms_norm_hidden_in_place(
                &mut encoder,
                &self.final_norm_handle,
                rms_eps,
                &request.token_state,
            )?;

            // LM Head GEMV
            self.executor.encode_dense_gemv_hidden_to_scratch(
                &mut encoder,
                &self.lm_head_handle,
                &request.token_state,
                &request.logits_scratch,
            )?;

            // GPU Greedy Argmax
            self.executor.encode_greedy_argmax(
                &mut encoder,
                &request.logits_scratch,
                &request.sampled_token_buf,
                &request.token_state,
                self.model_geometry.vocab_size,
            )?;

            // Copy final status and sampled token
            let final_status_offset = (num_layers * 4 + num_layers * top_k * 4) as u64;
            encoder.copy_buffer_to_buffer(
                request.token_state.status_buffer(),
                0,
                &request.staging_buffer,
                final_status_offset,
                4,
            );
            let token_offset = final_status_offset + 4;
            encoder.copy_buffer_to_buffer(
                request.sampled_token_buf.buffer(),
                0,
                &request.staging_buffer,
                token_offset,
                4,
            );
        } else {
            // Ingest-only: copy final status to staging
            let final_status_offset = (num_layers * 4 + num_layers * top_k * 4) as u64;
            encoder.copy_buffer_to_buffer(
                request.token_state.status_buffer(),
                0,
                &request.staging_buffer,
                final_status_offset,
                4,
            );
        }

        // ONE submission
        gpu.queue.submit(Some(encoder.finish()));

        // ONE map/readback
        let slice = request
            .staging_buffer
            .slice(..self.report_layout.total_bytes);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuNativeTokenLoopError::MapFailed(e.to_string()))?
            .map_err(|e| GpuNativeTokenLoopError::MapFailed(format!("{e:?}")))?;

        let mapped = slice.get_mapped_range();
        let report = self.report_layout.parse(&mapped)?;
        drop(mapped);
        request.staging_buffer.unmap();

        Ok(report)
    }
}

/// Request-local device resources for one GPU-native generation sequence.
pub struct GpuNativeRequestState {
    pub token_state: GpuNativeTokenState,
    pub kv_state: GpuNativeKvState,
    pub attn_scratch: GpuNativeAttentionScratch,
    pub router_scratch: GpuNativeRouterScratch,
    pub expert_scratch: GpuNativeQ4ExpertScratch,
    pub logits_scratch: GpuNativeScratch,
    pub sampled_token_buf: GpuNativeScratch,
    pub staging_buffer: wgpu::Buffer,
    pub committed_position: usize,
    pub max_seq_len: usize,
}

impl GpuNativeRequestState {
    pub fn committed_position(&self) -> usize {
        self.committed_position
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::architecture::Architecture;
    use crate::backend::gpu_native::{
        GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE, GPU_NATIVE_STATUS_EXPERT_NUMERICAL_FAILURE,
        GPU_NATIVE_STATUS_LM_HEAD_NUMERICAL_FAILURE, GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE,
    };
    use crate::config::Config;
    use crate::dense_tensor::DenseWeight;
    use crate::gating::{LinearGate, ScoringFunc};
    use crate::model::RealModelConfig;
    use crate::transformer::{LMHead, MultiHeadSelfAttention, RmsNorm, TransformerLayer};

    pub(crate) fn make_test_qwen3_moe_config() -> RealModelConfig {
        RealModelConfig {
            d_model: 32,
            d_ff: 32,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 32,
            num_layers: 2,
            num_experts: 4,
            top_k: 2,
            rope_base: 10_000.0,
            rms_eps: 1e-5,
            window_size: None,
            architecture: Architecture::Qwen3Moe,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        }
    }

    pub(crate) fn make_test_qwen3_moe_model(cfg: RealModelConfig) -> RealModel {
        let embedding = DenseWeight::from_f32(
            vec![0.1; cfg.vocab_size * cfg.d_model],
            cfg.vocab_size,
            cfg.d_model,
        );
        let lm_head = LMHead::new(
            vec![0.1; cfg.vocab_size * cfg.d_model],
            cfg.vocab_size,
            cfg.d_model,
        );
        let final_rms = RmsNorm::new(vec![1.0; cfg.d_model], cfg.rms_eps);

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            let attn = MultiHeadSelfAttention {
                d_model: cfg.d_model,
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                rope_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                attention_value_scale: None,
                rope_base: cfg.rope_base,
                wq: DenseWeight::from_f32(
                    vec![0.01; cfg.num_heads * cfg.head_dim * cfg.d_model],
                    cfg.num_heads * cfg.head_dim,
                    cfg.d_model,
                ),
                wk: DenseWeight::from_f32(
                    vec![0.01; cfg.num_kv_heads * cfg.head_dim * cfg.d_model],
                    cfg.num_kv_heads * cfg.head_dim,
                    cfg.d_model,
                ),
                wv: DenseWeight::from_f32(
                    vec![0.01; cfg.num_kv_heads * cfg.head_dim * cfg.d_model],
                    cfg.num_kv_heads * cfg.head_dim,
                    cfg.d_model,
                ),
                wo: DenseWeight::from_f32(
                    vec![0.01; cfg.d_model * cfg.num_heads * cfg.head_dim],
                    cfg.d_model,
                    cfg.num_heads * cfg.head_dim,
                ),
                window_size: None,
                q_norm: Some(RmsNorm::new(vec![1.0; cfg.head_dim], cfg.rms_eps)),
                k_norm: Some(RmsNorm::new(vec![1.0; cfg.head_dim], cfg.rms_eps)),
                rope_yarn: None,
                rope_cache: None,
                bq: None,
                bk: None,
                bv: None,
                bo: None,
                sink_bias: None,
            };
            let gate = LinearGate::new(
                vec![0.1; cfg.num_experts * cfg.d_model],
                cfg.num_experts,
                cfg.d_model,
                cfg.top_k,
            );
            let layer = TransformerLayer {
                rms_attn: RmsNorm::new(vec![1.0; cfg.d_model], cfg.rms_eps),
                attn,
                mla: None,
                rms_moe: RmsNorm::new(vec![1.0; cfg.d_model], cfg.rms_eps),
                gate,
                shared_expert: None,
                dense_ffn: None,
            };
            layers.push(layer);
        }

        RealModel {
            config: cfg,
            embedding,
            layers,
            final_rms,
            lm_head,
            load_status: crate::model::WeightLoadStatus::default(),
        }
    }

    #[test]
    fn compatibility_validation_accepts_supported_qwen3_moe() {
        let cfg = make_test_qwen3_moe_config();
        let model = make_test_qwen3_moe_model(cfg);
        assert!(GpuNativeTokenLoop::validate_model_compatibility(&model).is_ok());
    }

    #[test]
    fn compatibility_validation_rejects_wrong_architecture() {
        let mut cfg = make_test_qwen3_moe_config();
        cfg.architecture = Architecture::Mixtral;
        let model = make_test_qwen3_moe_model(cfg);
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model),
            Err(GpuNativeModelCompatibilityError::UnsupportedArchitecture { .. })
        ));
    }

    #[test]
    fn compatibility_validation_rejects_dense_and_shared_experts() {
        let cfg = make_test_qwen3_moe_config();
        let mut model = make_test_qwen3_moe_model(cfg);
        model.layers[0].dense_ffn = Some(
            crate::transformer::SharedExpert::from_projections(
                32,
                32,
                &vec![0.1; 32 * 32],
                &vec![0.1; 32 * 32],
                &vec![0.1; 32 * 32],
                None,
            )
            .unwrap(),
        );
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model),
            Err(GpuNativeModelCompatibilityError::DenseLayerUnsupported { .. })
        ));

        let cfg2 = make_test_qwen3_moe_config();
        let mut model2 = make_test_qwen3_moe_model(cfg2);
        model2.layers[0].shared_expert = Some(
            crate::transformer::SharedExpert::from_projections(
                32,
                32,
                &vec![0.1; 32 * 32],
                &vec![0.1; 32 * 32],
                &vec![0.1; 32 * 32],
                None,
            )
            .unwrap(),
        );
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model2),
            Err(GpuNativeModelCompatibilityError::SharedExpertUnsupported { .. })
        ));
    }

    #[test]
    fn compatibility_validation_rejects_asymmetric_v_and_sinks() {
        let cfg = make_test_qwen3_moe_config();
        let mut model = make_test_qwen3_moe_model(cfg);
        model.layers[0].attn.v_head_dim = 32;
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model),
            Err(GpuNativeModelCompatibilityError::AsymmetricVHeadDim { .. })
        ));

        let cfg2 = make_test_qwen3_moe_config();
        let mut model2 = make_test_qwen3_moe_model(cfg2);
        model2.layers[0].attn.sink_bias = Some(vec![0.0; 2]);
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model2),
            Err(GpuNativeModelCompatibilityError::AttentionSinkUnsupported { .. })
        ));
    }

    #[test]
    fn compatibility_validation_rejects_biases_and_sliding_window() {
        let cfg = make_test_qwen3_moe_config();
        let mut model = make_test_qwen3_moe_model(cfg);
        model.layers[0].attn.bq = Some(vec![0.0; 32]);
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model),
            Err(GpuNativeModelCompatibilityError::AttentionBiasesUnsupported { .. })
        ));

        let mut cfg2 = make_test_qwen3_moe_config();
        cfg2.window_size = Some(4096);
        let model2 = make_test_qwen3_moe_model(cfg2);
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model2),
            Err(GpuNativeModelCompatibilityError::SlidingWindowUnsupported { .. })
        ));
    }

    #[test]
    fn compatibility_validation_rejects_non_softmax_and_router_features() {
        let cfg = make_test_qwen3_moe_config();
        let mut model = make_test_qwen3_moe_model(cfg);
        model.layers[0].gate.scoring_func = ScoringFunc::Sigmoid;
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model),
            Err(GpuNativeModelCompatibilityError::NonSoftmaxRouter { .. })
        ));

        let cfg2 = make_test_qwen3_moe_config();
        let mut model2 = make_test_qwen3_moe_model(cfg2);
        model2.layers[0].gate.correction_bias = Some(vec![0.0; 4]);
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model2),
            Err(GpuNativeModelCompatibilityError::RouterCorrectionBiasUnsupported { .. })
        ));

        let cfg3 = make_test_qwen3_moe_config();
        let mut model3 = make_test_qwen3_moe_model(cfg3);
        model3.layers[0].gate.n_group = 2;
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model3),
            Err(GpuNativeModelCompatibilityError::GroupedRoutingUnsupported { .. })
        ));

        let cfg4 = make_test_qwen3_moe_config();
        let mut model4 = make_test_qwen3_moe_model(cfg4);
        model4.layers[0].gate.routed_scaling_factor = 2.0;
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model4),
            Err(GpuNativeModelCompatibilityError::RoutedScalingFactorUnsupported { .. })
        ));
    }

    #[test]
    fn boundary_report_layout_and_parser_work_correctly() {
        let layout = GpuNativeBoundaryReportLayout::try_new(2, 2).unwrap();
        assert_eq!(layout.total_bytes, 32);
        assert_eq!(layout.layer_status_offset, 0);
        assert_eq!(layout.layer_status_bytes, 8);
        assert_eq!(layout.selected_ids_offset, 8);
        assert_eq!(layout.selected_ids_bytes, 16);
        assert_eq!(layout.final_status_offset, 24);
        assert_eq!(layout.sampled_token_offset, 28);

        let mut bytes = vec![0u8; 32];
        bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&3u32.to_le_bytes());
        bytes[16..20].copy_from_slice(&0u32.to_le_bytes());
        bytes[20..24].copy_from_slice(&1u32.to_le_bytes());
        bytes[24..28].copy_from_slice(&0u32.to_le_bytes());
        bytes[28..32].copy_from_slice(&17u32.to_le_bytes());

        let report = layout.parse(&bytes).unwrap();
        assert_eq!(report.layer_statuses, vec![0, 0]);
        assert_eq!(report.selected_ids, vec![vec![2, 3], vec![0, 1]]);
        assert_eq!(report.final_status, 0);
        assert_eq!(report.sampled_token, 17);
        assert_eq!(report.first_failure_layer(), None);
    }

    #[test]
    fn first_failure_layer_detects_first_nonzero_transition() {
        let layout = GpuNativeBoundaryReportLayout::try_new(4, 2).unwrap();
        let mut bytes = vec![0u8; layout.total_bytes as usize];
        bytes[8..12].copy_from_slice(&4u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&4u32.to_le_bytes());

        let report = layout.parse(&bytes).unwrap();
        assert_eq!(report.first_failure_layer(), Some(2));
    }

    #[test]
    fn greedy_argmax_reference_semantics() {
        let reference_argmax = |logits: &[f32]| -> Result<u32, &'static str> {
            if logits.is_empty() {
                return Err("empty logits");
            }
            let mut best_val = -f32::MAX;
            let mut best_idx = None;
            for (idx, &val) in logits.iter().enumerate() {
                if !val.is_finite() {
                    return Err("non-finite logit");
                }
                if best_idx.is_none() || val > best_val {
                    best_val = val;
                    best_idx = Some(idx as u32);
                }
            }
            best_idx.ok_or("no candidate")
        };

        // Unique max
        assert_eq!(reference_argmax(&[1.0, 5.0, 2.0, 3.0]).unwrap(), 1);

        // Tied max selects lower token id
        assert_eq!(reference_argmax(&[1.0, 5.0, 2.0, 5.0, 3.0]).unwrap(), 1);

        // All negative finite logits
        assert_eq!(reference_argmax(&[-10.0, -2.0, -5.0]).unwrap(), 1);

        // Non-finite logits fail
        assert!(reference_argmax(&[1.0, f32::NAN, 2.0]).is_err());
        assert!(reference_argmax(&[1.0, f32::INFINITY, 2.0]).is_err());
        assert!(reference_argmax(&[1.0, f32::NEG_INFINITY, 2.0]).is_err());
    }

    #[test]
    fn fatal_status_mask_and_retry_clear_invariants() {
        assert_eq!(
            GPU_NATIVE_STATUS_FATAL_MASK & GPU_NATIVE_STATUS_RETRYABLE_MASK,
            0
        );
        assert_eq!(
            crate::backend::gpu_native::status_after_retryable_clear(
                GPU_NATIVE_STATUS_RETRYABLE_MASK
            ),
            0
        );
        assert_eq!(
            crate::backend::gpu_native::status_after_retryable_clear(
                GPU_NATIVE_STATUS_FATAL_MASK | GPU_NATIVE_STATUS_RETRYABLE_MASK
            ),
            GPU_NATIVE_STATUS_FATAL_MASK
        );
    }

    #[test]
    fn gpu_native_token_state_hidden_and_residual_have_no_copy_src() {
        let usage = crate::backend::gpu_native::GpuNativeTokenStateLayout::tensor_usage();
        assert!(!usage.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!usage.contains(wgpu::BufferUsages::MAP_READ));
        assert!(!usage.contains(wgpu::BufferUsages::MAP_WRITE));
    }

    #[test]
    fn gpu_native_config_defaults_and_validation() {
        let toml_str = r#"
            [model]
            data_dir = "."
            num_layers = 2
            num_experts = 4
            top_k = 2
            expert_size = 4096
            d_model = 32
            d_ff = 32
            dtype = "q4_0"

            [server]
            bind = "127.0.0.1:8080"
            max_concurrent_requests = 1
            session_ttl_secs = 0

            [storage]
            block_align = 4096
            cache_slots = 16

            [gpu_cache]
            enabled = true
            vram_capacity_mb = 128

            [real_transformer]
            enabled = true
            compute_offload = "gpu"
            gpu_native = true
            strict_weights = true
            max_batch_size = 1
        "#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(cfg.real_transformer.gpu_native);
        assert_eq!(cfg.real_transformer.gpu_native_max_seq_len, 4096);
        assert!(cfg.validate().is_ok());

        let mut invalid_cfg = cfg.clone();
        invalid_cfg.server.max_concurrent_requests = 2;
        assert!(invalid_cfg.validate().is_err());

        let mut invalid_cfg2 = cfg.clone();
        invalid_cfg2.real_transformer.max_batch_size = 2;
        assert!(invalid_cfg2.validate().is_err());
    }

    #[test]
    fn compatibility_validation_rejects_mla_and_oversized() {
        let mut cfg = make_test_qwen3_moe_config();
        cfg.num_experts = 16;
        cfg.top_k = 9;
        let model = make_test_qwen3_moe_model(cfg);
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model),
            Err(GpuNativeModelCompatibilityError::InvalidTopK { .. })
        ));

        let mut cfg2 = make_test_qwen3_moe_config();
        cfg2.num_experts = 129;
        let model2 = make_test_qwen3_moe_model(cfg2);
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model2),
            Err(GpuNativeModelCompatibilityError::TooManyExperts { .. })
        ));
    }

    #[test]
    fn boundary_report_parser_identifies_failure_modes() {
        let layout = GpuNativeBoundaryReportLayout::try_new(2, 2).unwrap();
        let mut bytes = vec![0u8; 32];

        // Fatal attention
        bytes[0..4].copy_from_slice(&GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE.to_le_bytes());
        let report = layout.parse(&bytes).unwrap();
        assert_eq!(report.first_failure_layer(), Some(0));
        assert_eq!(
            report.layer_statuses[0] & GPU_NATIVE_STATUS_FATAL_MASK,
            GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE
        );

        // Fatal router
        bytes[0..4].copy_from_slice(&0u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE.to_le_bytes());
        let report2 = layout.parse(&bytes).unwrap();
        assert_eq!(report2.first_failure_layer(), Some(1));
        assert_eq!(
            report2.layer_statuses[1] & GPU_NATIVE_STATUS_FATAL_MASK,
            GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE
        );

        // Fatal expert
        bytes[4..8].copy_from_slice(&GPU_NATIVE_STATUS_EXPERT_NUMERICAL_FAILURE.to_le_bytes());
        let report3 = layout.parse(&bytes).unwrap();
        assert_eq!(report3.first_failure_layer(), Some(1));
        assert_eq!(
            report3.layer_statuses[1] & GPU_NATIVE_STATUS_FATAL_MASK,
            GPU_NATIVE_STATUS_EXPERT_NUMERICAL_FAILURE
        );

        // Fatal LM head
        bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
        bytes[24..28].copy_from_slice(&GPU_NATIVE_STATUS_LM_HEAD_NUMERICAL_FAILURE.to_le_bytes());
        let report4 = layout.parse(&bytes).unwrap();
        assert_eq!(report4.first_failure_layer(), None);
        assert_eq!(
            report4.final_status & GPU_NATIVE_STATUS_FATAL_MASK,
            GPU_NATIVE_STATUS_LM_HEAD_NUMERICAL_FAILURE
        );
    }

    #[test]
    fn local_to_global_conversion_and_validation() {
        let num_experts = 4u32;
        let layer_0_locals = vec![1u32, 2];
        let layer_0_globals: Vec<u32> = layer_0_locals
            .iter()
            .map(|&loc| 0 * num_experts + loc)
            .collect();
        assert_eq!(layer_0_globals, vec![1, 2]);

        let layer_1_locals = vec![0u32, 3];
        let layer_1_globals: Vec<u32> = layer_1_locals
            .iter()
            .map(|&loc| 1 * num_experts + loc)
            .collect();
        assert_eq!(layer_1_globals, vec![4, 7]);
    }

    #[test]
    fn disabled_mode_preserves_legacy_state() {
        let toml_str = r#"
            [model]
            data_dir = "."
            num_layers = 2
            num_experts = 4
            top_k = 2
            expert_size = 4096
            d_model = 32
            d_ff = 32
            dtype = "q4_0"

            [server]
            bind = "127.0.0.1:8080"
            max_concurrent_requests = 4
            session_ttl_secs = 0

            [storage]
            block_align = 4096
            cache_slots = 16

            [gpu_cache]
            enabled = false
            vram_capacity_mb = 0

            [real_transformer]
            enabled = false
            gpu_native = false
        "#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(!cfg.real_transformer.gpu_native);
        assert!(cfg.validate().is_ok());
    }

    struct TempDir {
        path: std::path::PathBuf,
    }

    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("mer-test-{tag}-{id}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_full_token_loop_retry() {
        use crate::backend::{resolve_execution_context_for_gpu_native, GpuBackendGeometry};
        use crate::buffer_pool::BufferPool;
        use crate::expert_cache::{ExpertResident, GpuExpertCache, GpuResident};

        const D_MODEL: usize = 32;
        const D_FF: usize = 32;
        const NUM_EXPERTS: usize = 4;
        const TOP_K: usize = 2;
        const NUM_LAYERS: usize = 2;
        const VOCAB_SIZE: usize = 32;
        const MAX_SEQ_LEN: usize = 8;

        let geometry =
            GpuNativeQ4ExpertGeometry::try_new(D_MODEL, D_FF, NUM_EXPERTS, TOP_K).unwrap();
        let payload_template =
            crate::backend::gpu_native::tests::q4_uniform_expert(geometry, 0.01, 0.02, 0.005);
        let payload_bytes = payload_template.len();

        let gpu_cache = Arc::new(GpuExpertCache::new(payload_bytes * 16, 0.0, u64::MAX));

        let execution = resolve_execution_context_for_gpu_native(
            GpuBackendGeometry {
                num_layers: NUM_LAYERS,
                max_seq_len: MAX_SEQ_LEN,
                num_heads: 2,
                num_kv_heads: 2,
                head_dim: 16,
                v_head_dim: 16,
                q4_truncation_tolerance: 0,
            },
            gpu_cache.clone(),
        )
        .expect("L4 must construct the authoritative production GPU backend");

        let executor = Arc::new(
            execution
                .create_gpu_native_executor_context(D_MODEL)
                .expect("GPU-native executor must retain the authoritative backend"),
        );
        assert_eq!(executor.device_identity().vendor_id, 0x10de);
        assert!(
            executor.device_identity().name.contains("L4"),
            "ignored full token loop test must run only on an NVIDIA L4, got {}",
            executor.device_identity().name
        );

        let total_budget = (payload_bytes as u64) * (TOP_K as u64) * (NUM_LAYERS as u64) * 2;
        let residency_manager = Arc::new(
            GpuNativeTieredResidencyManager::try_new(
                executor.clone(),
                gpu_cache.clone(),
                NUM_LAYERS,
                geometry,
                total_budget,
            )
            .unwrap(),
        );

        let temp_dir = TempDir::new("full_token_loop");
        let storage = Arc::new(
            crate::io_provider::NvmeStorage::new(crate::io_provider::StorageConfig {
                base_path: temp_dir.path.clone(),
                expert_size: payload_bytes,
                block_align: 4,
                use_direct_io: false,
                num_experts_per_layer: Some(NUM_EXPERTS as u32),
            })
            .unwrap(),
        );

        let router = crate::gating::Router::Markov(Arc::new(crate::router::TopKRouter::new(
            (NUM_EXPERTS * NUM_LAYERS) as u32,
            TOP_K,
            0xC0FFEE,
        )));
        let predictor = Arc::new(crate::router::PredictiveLoader::new(
            (NUM_EXPERTS * NUM_LAYERS) as u32,
            TOP_K,
            0.0,
            42,
        ));

        let mut engine_builder = Engine::with_options_and_execution_context(
            Arc::new(
                crate::multi_layer_cache::MultiLayerExpertCache::with_uniform_capacity(
                    NUM_LAYERS,
                    16,
                    NUM_EXPERTS as u32,
                ),
            ),
            BufferPool::new(16, payload_bytes, 4),
            storage,
            router,
            predictor,
            crate::engine::ModelShape {
                d_model: D_MODEL,
                d_ff: D_FF,
                hidden_seed: 0xC0FFEE,
            },
            crate::engine::EngineOptions {
                io_only: false,
                dtype: crate::inference::WeightDtype::Q4_0,
                partial_load_fraction: 1.0,
                pin_after_observations: 0,
                use_qmm_for_q4: true,
                expert_execution_policy: crate::engine::ExpertExecutionPolicy::Auto,
                max_concurrent_prefetches: 64,
                max_fetch_yields: 128,
                prefetch_governor: false,
                prefetch_precision_floor: 0.0,
                prefetch_contention_weight: 0.0,
                cost_aware_eviction: false,
                pregate_enabled: false,
                collect_route_profile: false,
                policy: crate::inference::RealInferencePolicy::default(),
            },
            execution.clone(),
        );
        engine_builder
            .install_gpu_native_residency_manager(residency_manager.clone())
            .unwrap();
        let engine = Arc::new(engine_builder);

        let cfg = RealModelConfig {
            d_model: D_MODEL,
            d_ff: D_FF,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: VOCAB_SIZE,
            num_layers: NUM_LAYERS,
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
            rope_base: 10_000.0,
            rms_eps: 1e-5,
            window_size: None,
            architecture: Architecture::Qwen3Moe,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        };
        let model = make_test_qwen3_moe_model(cfg);

        let token_loop = GpuNativeTokenLoop::try_new(
            executor.clone(),
            residency_manager.clone(),
            &model,
            MAX_SEQ_LEN,
        )
        .unwrap();

        let mut request_state = token_loop.create_request_state().unwrap();

        let pool = BufferPool::new(16, payload_bytes, 4);
        for layer_idx in 0..NUM_LAYERS {
            for local_id in 0..NUM_EXPERTS as u32 {
                let global_id = layer_idx as u32 * NUM_EXPERTS as u32 + local_id;
                let payload = crate::backend::gpu_native::tests::q4_uniform_expert(
                    geometry,
                    0.01 * (global_id + 1) as f32,
                    0.02,
                    0.005,
                );
                let mut buffer = pool.try_acquire().expect("synthetic RAM slot");
                buffer.as_mut_slice().copy_from_slice(&payload);
                let resident = Arc::new(ExpertResident::new_with_block_align(global_id, buffer, 4));
                gpu_cache
                    .demand_admit_lru(Arc::new(GpuResident::new_with_dtype(
                        global_id,
                        payload.clone(),
                        crate::inference::WeightDtype::Q4_0,
                    )))
                    .unwrap();
                let admission = gpu_cache.current_admission(global_id).unwrap();
                let _ = (resident, admission);
            }
        }

        let first_token =
            pollster::block_on(token_loop.step_token(&engine, &mut request_state, 1, 0, true))
                .unwrap()
                .expect("cold token 0 must succeed after retries");

        assert_eq!(request_state.committed_position, 1);
        let snap1 = token_loop.snapshot();
        assert_eq!(snap1.tokens_completed, 1);
        assert_eq!(snap1.token_attempts, 3);
        assert_eq!(snap1.residency_miss_attempts, 2);
        assert_eq!(snap1.replay_attempts, 2);
        assert_eq!(snap1.residency_services, 2);
        assert_eq!(snap1.fatal_failures, 0);

        let _second_token = pollster::block_on(token_loop.step_token(
            &engine,
            &mut request_state,
            first_token,
            1,
            true,
        ))
        .unwrap()
        .expect("warm token 1 must succeed directly");

        assert_eq!(request_state.committed_position, 2);
        let snap2 = token_loop.snapshot();
        assert_eq!(snap2.tokens_completed, 2);
        assert_eq!(snap2.warm_tokens_completed, 1);
        assert_eq!(snap2.token_attempts, 4);
        assert_eq!(snap2.queue_submissions, 4);
        assert_eq!(snap2.boundary_maps, 4);
        assert_eq!(snap2.boundary_readbacks, 4);
        assert_eq!(snap2.residency_services, 2);
        assert_eq!(snap2.fatal_failures, 0);
    }
}
