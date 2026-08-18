//! Isolated foundations for a future GPU-native token loop.
//!
//! This module is not reachable from current operator-facing execution modes.
//! It owns request-local device state, persistent dense/RMSNorm model weights,
//! and encoder-only embedding, GEMV, RMSNorm, residual, attention-preparation,
//! causal-attention completion, and request-local KV primitives on the
//! authoritative WGPU device.
//! Later slices can compose those pieces without inheriting legacy host-shaped
//! upload/readback APIs.

// GPU-native slices remain intentionally unreachable from production token entrypoints.
#![allow(dead_code)]

use super::{create_startup_buffer, BackendBox, GpuDeviceIdentity, GpuStartupAllocationError};
use crate::dense_tensor::{DenseDType, DenseWeight};
use crate::inference::{Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};
use parking_lot::Mutex as ParkingMutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const GPU_NATIVE_STATUS_BYTES: u64 = std::mem::size_of::<u32>() as u64;
const GPU_NATIVE_DENSE_GEMV_SHADER: &str = include_str!("wgpu_shaders/gpu_native_dense_gemv.wgsl");
const GPU_NATIVE_EMBEDDING_SHADER: &str = include_str!("wgpu_shaders/gpu_native_embedding.wgsl");
const GPU_NATIVE_RMSNORM_SHADER: &str = include_str!("wgpu_shaders/gpu_native_rmsnorm.wgsl");
const GPU_NATIVE_ROPE_SHADER: &str = include_str!("wgpu_shaders/gpu_native_rope.wgsl");
const GPU_NATIVE_KV_APPEND_SHADER: &str = include_str!("wgpu_shaders/gpu_native_kv_append.wgsl");
const GPU_NATIVE_ATTENTION_SHADER: &str = include_str!("wgpu_shaders/gpu_native_attention.wgsl");
const GPU_NATIVE_WORKGROUP_SIZE: u32 = 64;
const GPU_NATIVE_ATTENTION_WORKGROUP_SIZE: u32 = 32;

/// Typed, fail-closed construction failure for the GPU-native bootstrap.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeBootstrapError {
    GpuBackendUnavailable,
    DeviceLost {
        detail: String,
    },
    InvalidDModel,
    StateSizeOverflow {
        d_model: usize,
    },
    InvalidDenseWeightKey,
    InvalidDenseWeightShape {
        rows: usize,
        cols: usize,
    },
    DenseWeightShapeOverflow {
        rows: usize,
        cols: usize,
    },
    DenseWeightDimensionTooLarge {
        rows: usize,
        cols: usize,
    },
    DenseWeightRowExceedsDeviceLimit {
        kind: GpuNativeDenseWeightKind,
        cols: usize,
        required: u64,
        maximum: u64,
    },
    DenseWeightByteLength {
        kind: GpuNativeDenseWeightKind,
        rows: usize,
        cols: usize,
        expected: usize,
        actual: usize,
    },
    DuplicateDenseWeight {
        key: String,
    },
    MissingDenseWeight {
        key: String,
    },
    ForeignDenseWeightHandle,
    StaleDenseWeightHandle {
        key: String,
    },
    DenseWeightKindMismatch {
        key: String,
        expected: GpuNativeDenseWeightKind,
        actual: GpuNativeDenseWeightKind,
    },
    DenseWeightShapeMismatch {
        key: String,
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    InvalidRmsNormWeightWidth {
        width: usize,
    },
    InvalidRmsNormEpsilon {
        epsilon_bits: u32,
    },
    ForeignRmsNormHandle,
    StaleRmsNormHandle {
        key: String,
    },
    RmsNormWeightWidth {
        expected: usize,
        actual: usize,
    },
    InvalidRmsNormGroups {
        groups: usize,
    },
    InvalidRmsNormGroupWidth {
        group_width: usize,
    },
    RmsNormGeometryOverflow {
        groups: usize,
        group_width: usize,
    },
    RmsNormScratchGeometry {
        expected: usize,
        actual: usize,
    },
    InvalidScratchElements,
    ScratchSizeOverflow {
        elements: usize,
    },
    ForeignTokenState,
    ForeignScratch,
    AliasedInputOutput,
    GemvInputLength {
        expected: usize,
        actual: usize,
    },
    GemvOutputLength {
        expected: usize,
        actual: usize,
    },
    InvalidEmbeddingToken {
        token_id: u32,
        vocab_size: usize,
    },
    EmbeddingWidth {
        expected: usize,
        actual: usize,
    },
    ResidualContributionWidth {
        expected: usize,
        actual: usize,
    },
    InvalidAttentionHeadCount {
        tensor: GpuNativeAttentionTensor,
        heads: usize,
    },
    InvalidAttentionHeadGeometry {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    },
    AttentionGeometryOverflow {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    },
    AttentionDModelMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidRopeDimension {
        rope_dim: usize,
        head_dim: usize,
    },
    OddRopeDimension {
        rope_dim: usize,
    },
    InvalidRopeBase {
        base_bits: u32,
    },
    InvalidRopeInverseFrequency {
        index: usize,
        value_bits: u32,
    },
    InvalidRopeAttentionFactor {
        factor_bits: u32,
    },
    ForeignRopeHandle,
    StaleRopeHandle {
        key: String,
    },
    RopeDimensionMismatch {
        expected: usize,
        actual: usize,
    },
    RopeParameterWidth {
        expected: usize,
        actual: usize,
    },
    ForeignAttentionPlan,
    AttentionPlanLayerOutOfRange {
        layer_index: usize,
        num_layers: usize,
    },
    ForeignAttentionScratch,
    AttentionScratchGeometry {
        expected: GpuNativeAttentionGeometry,
        actual: GpuNativeAttentionGeometry,
    },
    AttentionScratchWidth {
        tensor: GpuNativeAttentionTensor,
        expected: usize,
        actual: usize,
    },
    AttentionProjectionShape {
        tensor: GpuNativeAttentionTensor,
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    AttentionNormWidth {
        tensor: GpuNativeAttentionTensor,
        expected: usize,
        actual: usize,
    },
    InvalidKvLayerCount,
    InvalidKvCapacity,
    InvalidKvWidth,
    KvCapacityOverflow {
        num_layers: usize,
        max_seq_len: usize,
        kv_width: usize,
    },
    KvBufferLimit {
        required: u64,
        max_buffer_size: u64,
        max_storage_binding_size: u64,
    },
    ForeignKvState,
    InvalidKvLayer {
        layer: usize,
        num_layers: usize,
    },
    InvalidKvPosition {
        position: usize,
        max_seq_len: usize,
    },
    AttentionSequenceLengthOverflow {
        position: usize,
    },
    InvalidAttentionSequenceLength {
        seq_len: usize,
        max_seq_len: usize,
    },
    KvWidth {
        expected: usize,
        actual: usize,
    },
    DispatchGeometryUnsupported {
        workgroups: u64,
        maximum: u32,
    },
    AttentionWorkgroupUnsupported {
        required: u32,
        max_size_x: u32,
        max_invocations: u32,
    },
    Allocation(GpuStartupAllocationError),
}

impl fmt::Display for GpuNativeBootstrapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GpuBackendUnavailable => write!(
                f,
                "GPU-native execution requires the authoritative production GPU backend"
            ),
            Self::DeviceLost { detail } => {
                write!(
                    f,
                    "GPU device was lost before GPU-native bootstrap: {detail}"
                )
            }
            Self::InvalidDModel => write!(f, "GPU-native d_model must be non-zero"),
            Self::StateSizeOverflow { d_model } => write!(
                f,
                "GPU-native token-state size overflows for d_model={d_model}"
            ),
            Self::InvalidDenseWeightKey => {
                write!(f, "GPU-native dense weight key must be non-empty")
            }
            Self::InvalidDenseWeightShape { rows, cols } => write!(
                f,
                "GPU-native dense weight shape [{rows}, {cols}] must be non-empty"
            ),
            Self::DenseWeightShapeOverflow { rows, cols } => write!(
                f,
                "GPU-native dense weight shape [{rows}, {cols}] overflows"
            ),
            Self::DenseWeightDimensionTooLarge { rows, cols } => write!(
                f,
                "GPU-native dense weight shape [{rows}, {cols}] exceeds u32 shader geometry"
            ),
            Self::DenseWeightRowExceedsDeviceLimit {
                kind,
                cols,
                required,
                maximum,
            } => write!(
                f,
                "GPU-native {kind:?} dense weight row with {cols} columns requires {required} bytes, exceeding the physical chunk limit {maximum}"
            ),
            Self::DenseWeightByteLength {
                kind,
                rows,
                cols,
                expected,
                actual,
            } => write!(
                f,
                "GPU-native {kind:?} dense weight [{rows}, {cols}] has {actual} bytes, expected {expected}"
            ),
            Self::DuplicateDenseWeight { key } => {
                write!(f, "GPU-native dense weight {key:?} is already registered")
            }
            Self::MissingDenseWeight { key } => {
                write!(f, "GPU-native dense weight {key:?} is not registered")
            }
            Self::ForeignDenseWeightHandle => write!(
                f,
                "GPU-native dense weight handle belongs to a different executor context"
            ),
            Self::StaleDenseWeightHandle { key } => {
                write!(f, "GPU-native dense weight handle for {key:?} is stale")
            }
            Self::DenseWeightKindMismatch {
                key,
                expected,
                actual,
            } => write!(
                f,
                "GPU-native dense weight {key:?} has kind {actual:?}, expected {expected:?}"
            ),
            Self::DenseWeightShapeMismatch {
                key,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                f,
                "GPU-native dense weight {key:?} has shape [{actual_rows}, {actual_cols}], expected [{expected_rows}, {expected_cols}]"
            ),
            Self::InvalidRmsNormWeightWidth { width } => write!(
                f,
                "GPU-native RMSNorm weight width must be non-zero, got {width}"
            ),
            Self::InvalidRmsNormEpsilon { epsilon_bits } => write!(
                f,
                "GPU-native RMSNorm epsilon must be finite and non-negative, got {}",
                f32::from_bits(*epsilon_bits)
            ),
            Self::ForeignRmsNormHandle => write!(
                f,
                "GPU-native RMSNorm handle belongs to a different executor context"
            ),
            Self::StaleRmsNormHandle { key } => {
                write!(f, "GPU-native RMSNorm handle for {key:?} is stale")
            }
            Self::RmsNormWeightWidth { expected, actual } => write!(
                f,
                "GPU-native RMSNorm weight width is {actual}, expected {expected}"
            ),
            Self::InvalidRmsNormGroups { groups } => write!(
                f,
                "GPU-native RMSNorm group count must be non-zero, got {groups}"
            ),
            Self::InvalidRmsNormGroupWidth { group_width } => write!(
                f,
                "GPU-native RMSNorm group width must be non-zero, got {group_width}"
            ),
            Self::RmsNormGeometryOverflow {
                groups,
                group_width,
            } => write!(
                f,
                "GPU-native RMSNorm geometry {groups} x {group_width} overflows"
            ),
            Self::RmsNormScratchGeometry { expected, actual } => write!(
                f,
                "GPU-native RMSNorm scratch has {actual} elements, expected {expected}"
            ),
            Self::InvalidScratchElements => {
                write!(f, "GPU-native scratch length must be non-zero")
            }
            Self::ScratchSizeOverflow { elements } => write!(
                f,
                "GPU-native scratch size overflows for {elements} elements"
            ),
            Self::ForeignTokenState => write!(
                f,
                "GPU-native token state belongs to a different executor context"
            ),
            Self::ForeignScratch => write!(
                f,
                "GPU-native scratch belongs to a different executor context"
            ),
            Self::AliasedInputOutput => {
                write!(f, "GPU-native GEMV input and output buffers must be distinct")
            }
            Self::GemvInputLength { expected, actual } => write!(
                f,
                "GPU-native GEMV input has {actual} elements, expected {expected}"
            ),
            Self::GemvOutputLength { expected, actual } => write!(
                f,
                "GPU-native GEMV output has {actual} elements, expected {expected}"
            ),
            Self::InvalidEmbeddingToken {
                token_id,
                vocab_size,
            } => write!(
                f,
                "GPU-native embedding token {token_id} is outside vocabulary size {vocab_size}"
            ),
            Self::EmbeddingWidth { expected, actual } => write!(
                f,
                "GPU-native embedding width is {actual}, expected token-state width {expected}"
            ),
            Self::ResidualContributionWidth { expected, actual } => write!(
                f,
                "GPU-native residual contribution has {actual} elements, expected {expected}"
            ),
            Self::InvalidAttentionHeadCount { tensor, heads } => write!(
                f,
                "GPU-native {tensor} head count must be non-zero, got {heads}"
            ),
            Self::InvalidAttentionHeadGeometry {
                num_heads,
                num_kv_heads,
                head_dim,
            } => write!(
                f,
                "GPU-native attention geometry requires non-zero head_dim and query heads divisible by KV heads, got num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}"
            ),
            Self::AttentionGeometryOverflow {
                num_heads,
                num_kv_heads,
                head_dim,
            } => write!(
                f,
                "GPU-native attention geometry overflows for num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}"
            ),
            Self::AttentionDModelMismatch { expected, actual } => write!(
                f,
                "GPU-native attention d_model is {actual}, expected executor width {expected}"
            ),
            Self::InvalidRopeDimension { rope_dim, head_dim } => write!(
                f,
                "GPU-native RoPE dimension must be in 1..={head_dim}, got {rope_dim}"
            ),
            Self::OddRopeDimension { rope_dim } => write!(
                f,
                "GPU-native RoPE dimension must be even, got {rope_dim}"
            ),
            Self::InvalidRopeBase { base_bits } => write!(
                f,
                "GPU-native RoPE base must be finite and positive, got {}",
                f32::from_bits(*base_bits)
            ),
            Self::InvalidRopeInverseFrequency { index, value_bits } => write!(
                f,
                "GPU-native RoPE inverse frequency {index} must be finite and positive, got {}",
                f32::from_bits(*value_bits)
            ),
            Self::InvalidRopeAttentionFactor { factor_bits } => write!(
                f,
                "GPU-native RoPE attention factor must be finite and positive, got {}",
                f32::from_bits(*factor_bits)
            ),
            Self::ForeignRopeHandle => write!(
                f,
                "GPU-native RoPE handle belongs to a different executor context"
            ),
            Self::StaleRopeHandle { key } => {
                write!(f, "GPU-native RoPE handle for {key:?} is stale")
            }
            Self::RopeDimensionMismatch { expected, actual } => write!(
                f,
                "GPU-native RoPE dimension is {actual}, expected {expected}"
            ),
            Self::RopeParameterWidth { expected, actual } => write!(
                f,
                "GPU-native RoPE inverse-frequency table has {actual} values, expected {expected}"
            ),
            Self::ForeignAttentionPlan => write!(
                f,
                "GPU-native attention plan belongs to a different executor context"
            ),
            Self::AttentionPlanLayerOutOfRange {
                layer_index,
                num_layers,
            } => write!(
                f,
                "GPU-native attention plan layer {layer_index} is outside request-local KV layer count {num_layers}"
            ),
            Self::ForeignAttentionScratch => write!(
                f,
                "GPU-native attention scratch belongs to a different executor context"
            ),
            Self::AttentionScratchGeometry { expected, actual } => write!(
                f,
                "GPU-native attention scratch geometry {actual:?} does not match plan geometry {expected:?}"
            ),
            Self::AttentionScratchWidth {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "GPU-native {tensor} scratch has {actual} elements, expected {expected}"
            ),
            Self::AttentionProjectionShape {
                tensor,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                f,
                "GPU-native {tensor} projection has shape [{actual_rows}, {actual_cols}], expected [{expected_rows}, {expected_cols}]"
            ),
            Self::AttentionNormWidth {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "GPU-native {tensor} norm gain has width {actual}, expected {expected}"
            ),
            Self::InvalidKvLayerCount => {
                write!(f, "GPU-native KV layer count must be non-zero")
            }
            Self::InvalidKvCapacity => {
                write!(f, "GPU-native KV maximum sequence length must be non-zero")
            }
            Self::InvalidKvWidth => write!(f, "GPU-native KV width must be non-zero"),
            Self::KvCapacityOverflow {
                num_layers,
                max_seq_len,
                kv_width,
            } => write!(
                f,
                "GPU-native KV capacity overflows for layers={num_layers}, max_seq_len={max_seq_len}, width={kv_width}"
            ),
            Self::KvBufferLimit {
                required,
                max_buffer_size,
                max_storage_binding_size,
            } => write!(
                f,
                "GPU-native per-layer KV buffer requires {required} bytes, exceeding max_buffer_size={max_buffer_size} or max_storage_buffer_binding_size={max_storage_binding_size}"
            ),
            Self::ForeignKvState => write!(
                f,
                "GPU-native KV state belongs to a different executor context"
            ),
            Self::InvalidKvLayer { layer, num_layers } => write!(
                f,
                "GPU-native KV layer {layer} is outside layer count {num_layers}"
            ),
            Self::InvalidKvPosition {
                position,
                max_seq_len,
            } => write!(
                f,
                "GPU-native KV position {position} is outside capacity {max_seq_len}"
            ),
            Self::AttentionSequenceLengthOverflow { position } => write!(
                f,
                "GPU-native causal attention sequence length overflows for position {position}"
            ),
            Self::InvalidAttentionSequenceLength {
                seq_len,
                max_seq_len,
            } => write!(
                f,
                "GPU-native causal attention sequence length {seq_len} is outside 1..={max_seq_len}"
            ),
            Self::KvWidth { expected, actual } => write!(
                f,
                "GPU-native KV width is {actual}, expected {expected}"
            ),
            Self::DispatchGeometryUnsupported {
                workgroups,
                maximum,
            } => write!(
                f,
                "GPU-native dispatch requires {workgroups} workgroups, exceeding device maximum {maximum}"
            ),
            Self::AttentionWorkgroupUnsupported {
                required,
                max_size_x,
                max_invocations,
            } => write!(
                f,
                "GPU-native causal attention requires a {required}-lane workgroup, exceeding max_compute_workgroup_size_x={max_size_x} or max_compute_invocations_per_workgroup={max_invocations}"
            ),
            Self::Allocation(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for GpuNativeBootstrapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GpuStartupAllocationError> for GpuNativeBootstrapError {
    fn from(error: GpuStartupAllocationError) -> Self {
        Self::Allocation(error)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeAttentionTensor {
    Query,
    Key,
    Value,
    Context,
    Output,
}

impl fmt::Display for GpuNativeAttentionTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Query => write!(f, "query"),
            Self::Key => write!(f, "key"),
            Self::Value => write!(f, "value"),
            Self::Context => write!(f, "attention context"),
            Self::Output => write!(f, "attention output"),
        }
    }
}

/// Dense storage identity understood by the GPU-native shaders.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeDenseWeightKind {
    F32,
    Q8_0,
}

impl From<DenseDType> for GpuNativeDenseWeightKind {
    fn from(dtype: DenseDType) -> Self {
        match dtype {
            DenseDType::F32 => Self::F32,
            DenseDType::Q8_0 => Self::Q8_0,
        }
    }
}

/// Checked immutable matrix layout. `payload_bytes` is the exact model
/// payload; `allocation_bytes` includes only the trailing zero padding WGPU
/// requires to make a storage-buffer write four-byte aligned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeDenseWeightLayout {
    kind: GpuNativeDenseWeightKind,
    rows: usize,
    cols: usize,
    payload_bytes: u64,
    allocation_bytes: u64,
}

impl GpuNativeDenseWeightLayout {
    fn try_new(
        kind: GpuNativeDenseWeightKind,
        rows: usize,
        cols: usize,
        actual_bytes: usize,
    ) -> Result<Self, GpuNativeBootstrapError> {
        if rows == 0 || cols == 0 {
            return Err(GpuNativeBootstrapError::InvalidDenseWeightShape { rows, cols });
        }
        let elements = rows
            .checked_mul(cols)
            .ok_or(GpuNativeBootstrapError::DenseWeightShapeOverflow { rows, cols })?;
        if u32::try_from(rows).is_err()
            || u32::try_from(cols).is_err()
            || u32::try_from(elements).is_err()
        {
            return Err(GpuNativeBootstrapError::DenseWeightDimensionTooLarge { rows, cols });
        }
        let expected_bytes = match kind {
            GpuNativeDenseWeightKind::F32 => elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(GpuNativeBootstrapError::DenseWeightShapeOverflow { rows, cols })?,
            GpuNativeDenseWeightKind::Q8_0 => elements
                .div_ceil(Q8_0_BLOCK_ELEMS)
                .checked_mul(Q8_0_BLOCK_BYTES)
                .ok_or(GpuNativeBootstrapError::DenseWeightShapeOverflow { rows, cols })?,
        };
        if actual_bytes != expected_bytes {
            return Err(GpuNativeBootstrapError::DenseWeightByteLength {
                kind,
                rows,
                cols,
                expected: expected_bytes,
                actual: actual_bytes,
            });
        }
        // Q8 byte extraction also uses u32 offsets in WGSL. F32 indexes an
        // array<u32> by element, so its already-checked element count is the
        // relevant bound rather than its four-times-larger byte count.
        if kind == GpuNativeDenseWeightKind::Q8_0 && u32::try_from(expected_bytes).is_err() {
            return Err(GpuNativeBootstrapError::DenseWeightDimensionTooLarge { rows, cols });
        }
        let allocation_bytes = expected_bytes
            .checked_add(3)
            .map(|bytes| bytes & !3)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(GpuNativeBootstrapError::DenseWeightShapeOverflow { rows, cols })?;
        let payload_bytes = u64::try_from(expected_bytes)
            .map_err(|_| GpuNativeBootstrapError::DenseWeightShapeOverflow { rows, cols })?;
        Ok(Self {
            kind,
            rows,
            cols,
            payload_bytes,
            allocation_bytes,
        })
    }

    fn from_weight(weight: &DenseWeight) -> Result<Self, GpuNativeBootstrapError> {
        Self::try_new(
            weight.dtype().into(),
            weight.rows(),
            weight.cols(),
            weight.resident_bytes(),
        )
    }

    pub(crate) const fn kind(self) -> GpuNativeDenseWeightKind {
        self.kind
    }

    pub(crate) const fn rows(self) -> usize {
        self.rows
    }

    pub(crate) const fn cols(self) -> usize {
        self.cols
    }

    pub(crate) const fn payload_bytes(self) -> u64 {
        self.payload_bytes
    }

    pub(crate) const fn allocation_bytes(self) -> u64 {
        self.allocation_bytes
    }

    fn usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
    }

    fn validate_for_limits(
        self,
        label: &str,
        limits: &wgpu::Limits,
    ) -> Result<(), GpuNativeBootstrapError> {
        super::validate_startup_buffer(label, self.allocation_bytes, Self::usage(), limits)?;
        Ok(())
    }

    fn validate_embedding_token(self, token_id: u32) -> Result<(), GpuNativeBootstrapError> {
        if token_id as usize >= self.rows {
            return Err(GpuNativeBootstrapError::InvalidEmbeddingToken {
                token_id,
                vocab_size: self.rows,
            });
        }
        Ok(())
    }
}

/// One independently bindable physical buffer covering complete logical rows.
/// Q8_0 chunks retain the source tensor's global flat-block convention and can
/// therefore duplicate the single boundary block shared by adjacent chunks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GpuNativeDenseWeightChunkPlan {
    row_start: usize,
    row_count: usize,
    first_block: usize,
    payload_offset_bytes: usize,
    payload_bytes: u64,
    allocation_bytes: u64,
}

impl GpuNativeDenseWeightChunkPlan {
    fn row_end(self) -> usize {
        self.row_start + self.row_count
    }

    fn contains_row(self, row: usize) -> bool {
        self.row_start <= row && row < self.row_end()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GpuNativeDenseWeightPlan {
    layout: GpuNativeDenseWeightLayout,
    chunks: Vec<GpuNativeDenseWeightChunkPlan>,
    physical_allocation_bytes: u64,
}

impl GpuNativeDenseWeightPlan {
    fn try_new(
        layout: GpuNativeDenseWeightLayout,
        limits: &wgpu::Limits,
    ) -> Result<Self, GpuNativeBootstrapError> {
        let maximum = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));
        let mut chunks = Vec::new();
        let mut row_start = 0usize;
        let mut physical_allocation_bytes = 0u64;

        while row_start < layout.rows {
            let remaining = layout.rows - row_start;
            let row_count = match layout.kind {
                GpuNativeDenseWeightKind::F32 => {
                    let row_bytes = layout.cols.checked_mul(std::mem::size_of::<f32>()).ok_or(
                        GpuNativeBootstrapError::DenseWeightShapeOverflow {
                            rows: layout.rows,
                            cols: layout.cols,
                        },
                    )?;
                    let row_bytes = u64::try_from(row_bytes).map_err(|_| {
                        GpuNativeBootstrapError::DenseWeightShapeOverflow {
                            rows: layout.rows,
                            cols: layout.cols,
                        }
                    })?;
                    if row_bytes > maximum {
                        return Err(GpuNativeBootstrapError::DenseWeightRowExceedsDeviceLimit {
                            kind: layout.kind,
                            cols: layout.cols,
                            required: row_bytes,
                            maximum,
                        });
                    }
                    remaining.min(usize::try_from(maximum / row_bytes).unwrap_or(usize::MAX))
                }
                GpuNativeDenseWeightKind::Q8_0 => {
                    let one_row = Self::q8_chunk(layout, row_start, 1)?;
                    if one_row.allocation_bytes > maximum {
                        return Err(GpuNativeBootstrapError::DenseWeightRowExceedsDeviceLimit {
                            kind: layout.kind,
                            cols: layout.cols,
                            required: one_row.allocation_bytes,
                            maximum,
                        });
                    }
                    let mut low = 1usize;
                    let mut high = remaining;
                    while low < high {
                        let middle = low + (high - low).div_ceil(2);
                        if Self::q8_chunk(layout, row_start, middle)?.allocation_bytes <= maximum {
                            low = middle;
                        } else {
                            high = middle - 1;
                        }
                    }
                    low
                }
            };

            let chunk = match layout.kind {
                GpuNativeDenseWeightKind::F32 => {
                    let row_bytes = layout.cols * std::mem::size_of::<f32>();
                    let payload_offset_bytes = row_start * row_bytes;
                    let payload_bytes = u64::try_from(row_count * row_bytes).map_err(|_| {
                        GpuNativeBootstrapError::DenseWeightShapeOverflow {
                            rows: layout.rows,
                            cols: layout.cols,
                        }
                    })?;
                    GpuNativeDenseWeightChunkPlan {
                        row_start,
                        row_count,
                        first_block: 0,
                        payload_offset_bytes,
                        payload_bytes,
                        allocation_bytes: payload_bytes,
                    }
                }
                GpuNativeDenseWeightKind::Q8_0 => Self::q8_chunk(layout, row_start, row_count)?,
            };
            physical_allocation_bytes = physical_allocation_bytes
                .checked_add(chunk.allocation_bytes)
                .ok_or(GpuNativeBootstrapError::DenseWeightShapeOverflow {
                    rows: layout.rows,
                    cols: layout.cols,
                })?;
            chunks.push(chunk);
            row_start += row_count;
        }

        Ok(Self {
            layout,
            chunks,
            physical_allocation_bytes,
        })
    }

    fn q8_chunk(
        layout: GpuNativeDenseWeightLayout,
        row_start: usize,
        row_count: usize,
    ) -> Result<GpuNativeDenseWeightChunkPlan, GpuNativeBootstrapError> {
        let element_start = row_start.checked_mul(layout.cols).ok_or(
            GpuNativeBootstrapError::DenseWeightShapeOverflow {
                rows: layout.rows,
                cols: layout.cols,
            },
        )?;
        let element_end = row_start
            .checked_add(row_count)
            .and_then(|row_end| row_end.checked_mul(layout.cols))
            .ok_or(GpuNativeBootstrapError::DenseWeightShapeOverflow {
                rows: layout.rows,
                cols: layout.cols,
            })?;
        let first_block = element_start / Q8_0_BLOCK_ELEMS;
        let block_end = element_end.div_ceil(Q8_0_BLOCK_ELEMS);
        let block_count = block_end - first_block;
        let payload_offset_bytes = first_block.checked_mul(Q8_0_BLOCK_BYTES).ok_or(
            GpuNativeBootstrapError::DenseWeightShapeOverflow {
                rows: layout.rows,
                cols: layout.cols,
            },
        )?;
        let payload_bytes_usize = block_count.checked_mul(Q8_0_BLOCK_BYTES).ok_or(
            GpuNativeBootstrapError::DenseWeightShapeOverflow {
                rows: layout.rows,
                cols: layout.cols,
            },
        )?;
        let allocation_bytes_usize = payload_bytes_usize
            .checked_add(3)
            .map(|bytes| bytes & !3)
            .ok_or(GpuNativeBootstrapError::DenseWeightShapeOverflow {
                rows: layout.rows,
                cols: layout.cols,
            })?;
        Ok(GpuNativeDenseWeightChunkPlan {
            row_start,
            row_count,
            first_block,
            payload_offset_bytes,
            payload_bytes: payload_bytes_usize as u64,
            allocation_bytes: allocation_bytes_usize as u64,
        })
    }
}

/// Checked logical grouping for one RMSNorm dispatch. The shader launches one
/// workgroup per group and reuses one `group_width`-element F32 gain vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GpuNativeRmsNormGeometry {
    groups: usize,
    group_width: usize,
    elements: usize,
}

impl GpuNativeRmsNormGeometry {
    fn try_new(
        groups: usize,
        group_width: usize,
        actual_elements: usize,
        weight_width: usize,
    ) -> Result<Self, GpuNativeBootstrapError> {
        if groups == 0 {
            return Err(GpuNativeBootstrapError::InvalidRmsNormGroups { groups });
        }
        if group_width == 0 {
            return Err(GpuNativeBootstrapError::InvalidRmsNormGroupWidth { group_width });
        }
        let elements = groups.checked_mul(group_width).ok_or(
            GpuNativeBootstrapError::RmsNormGeometryOverflow {
                groups,
                group_width,
            },
        )?;
        if u32::try_from(groups).is_err()
            || u32::try_from(group_width).is_err()
            || u32::try_from(elements).is_err()
        {
            return Err(GpuNativeBootstrapError::RmsNormGeometryOverflow {
                groups,
                group_width,
            });
        }
        if actual_elements != elements {
            return Err(GpuNativeBootstrapError::RmsNormScratchGeometry {
                expected: elements,
                actual: actual_elements,
            });
        }
        if weight_width != group_width {
            return Err(GpuNativeBootstrapError::RmsNormWeightWidth {
                expected: group_width,
                actual: weight_width,
            });
        }
        Ok(Self {
            groups,
            group_width,
            elements,
        })
    }

    fn checked_workgroups(self, limits: &wgpu::Limits) -> Result<u32, GpuNativeBootstrapError> {
        let workgroups = self.groups as u64;
        if workgroups > limits.max_compute_workgroups_per_dimension as u64 {
            return Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
                workgroups,
                maximum: limits.max_compute_workgroups_per_dimension,
            });
        }
        Ok(self.groups as u32)
    }
}

/// Qwen-compatible GPU-native attention geometry. Query heads may use GQA,
/// but Q/K/V share one head width and V is deliberately not asymmetric.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeAttentionGeometry {
    d_model: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    q_width: usize,
    kv_width: usize,
}

impl GpuNativeAttentionGeometry {
    pub(crate) fn try_new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_dim: usize,
    ) -> Result<Self, GpuNativeBootstrapError> {
        if num_heads == 0 {
            return Err(GpuNativeBootstrapError::InvalidAttentionHeadCount {
                tensor: GpuNativeAttentionTensor::Query,
                heads: num_heads,
            });
        }
        if num_kv_heads == 0 {
            return Err(GpuNativeBootstrapError::InvalidAttentionHeadCount {
                tensor: GpuNativeAttentionTensor::Key,
                heads: num_kv_heads,
            });
        }
        if head_dim == 0 || num_heads < num_kv_heads || num_heads % num_kv_heads != 0 {
            return Err(GpuNativeBootstrapError::InvalidAttentionHeadGeometry {
                num_heads,
                num_kv_heads,
                head_dim,
            });
        }
        validate_rope_dimension(rope_dim, head_dim)?;
        let q_width = num_heads.checked_mul(head_dim).ok_or(
            GpuNativeBootstrapError::AttentionGeometryOverflow {
                num_heads,
                num_kv_heads,
                head_dim,
            },
        )?;
        let kv_width = num_kv_heads.checked_mul(head_dim).ok_or(
            GpuNativeBootstrapError::AttentionGeometryOverflow {
                num_heads,
                num_kv_heads,
                head_dim,
            },
        )?;
        if d_model == 0
            || u32::try_from(d_model).is_err()
            || u32::try_from(q_width).is_err()
            || u32::try_from(kv_width).is_err()
            || u32::try_from(head_dim).is_err()
            || u32::try_from(rope_dim).is_err()
        {
            return Err(GpuNativeBootstrapError::AttentionGeometryOverflow {
                num_heads,
                num_kv_heads,
                head_dim,
            });
        }
        Ok(Self {
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim,
            q_width,
            kv_width,
        })
    }

    pub(crate) const fn d_model(self) -> usize {
        self.d_model
    }

    pub(crate) const fn num_heads(self) -> usize {
        self.num_heads
    }

    pub(crate) const fn num_kv_heads(self) -> usize {
        self.num_kv_heads
    }

    pub(crate) const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub(crate) const fn rope_dim(self) -> usize {
        self.rope_dim
    }

    pub(crate) const fn q_width(self) -> usize {
        self.q_width
    }

    pub(crate) const fn kv_width(self) -> usize {
        self.kv_width
    }
}

fn validate_causal_attention_dispatch(
    geometry: GpuNativeAttentionGeometry,
    limits: &wgpu::Limits,
) -> Result<u32, GpuNativeBootstrapError> {
    if limits.max_compute_workgroup_size_x < GPU_NATIVE_ATTENTION_WORKGROUP_SIZE
        || limits.max_compute_invocations_per_workgroup < GPU_NATIVE_ATTENTION_WORKGROUP_SIZE
    {
        return Err(GpuNativeBootstrapError::AttentionWorkgroupUnsupported {
            required: GPU_NATIVE_ATTENTION_WORKGROUP_SIZE,
            max_size_x: limits.max_compute_workgroup_size_x,
            max_invocations: limits.max_compute_invocations_per_workgroup,
        });
    }
    if geometry.num_heads as u64 > limits.max_compute_workgroups_per_dimension as u64 {
        return Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
            workgroups: geometry.num_heads as u64,
            maximum: limits.max_compute_workgroups_per_dimension,
        });
    }
    Ok(geometry.num_heads as u32)
}

fn validate_rope_dimension(
    rope_dim: usize,
    head_dim: usize,
) -> Result<(), GpuNativeBootstrapError> {
    if rope_dim == 0 || rope_dim > head_dim {
        return Err(GpuNativeBootstrapError::InvalidRopeDimension { rope_dim, head_dim });
    }
    if !rope_dim.is_multiple_of(2) {
        return Err(GpuNativeBootstrapError::OddRopeDimension { rope_dim });
    }
    Ok(())
}

fn validate_rope_parameters(
    layout: GpuNativeRopeLayout,
    inverse_frequencies: &[f32],
    attention_factor: f32,
) -> Result<(), GpuNativeBootstrapError> {
    if inverse_frequencies.len() != layout.pairs {
        return Err(GpuNativeBootstrapError::RopeParameterWidth {
            expected: layout.pairs,
            actual: inverse_frequencies.len(),
        });
    }
    for (index, value) in inverse_frequencies.iter().copied().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(GpuNativeBootstrapError::InvalidRopeInverseFrequency {
                index,
                value_bits: value.to_bits(),
            });
        }
    }
    if !attention_factor.is_finite() || attention_factor <= 0.0 {
        return Err(GpuNativeBootstrapError::InvalidRopeAttentionFactor {
            factor_bits: attention_factor.to_bits(),
        });
    }
    Ok(())
}

fn standard_rope_inverse_frequencies(
    rope_dim: usize,
    base: f32,
) -> Result<Vec<f32>, GpuNativeBootstrapError> {
    let layout = GpuNativeRopeLayout::try_new(rope_dim, rope_dim)?;
    if !base.is_finite() || base <= 0.0 {
        return Err(GpuNativeBootstrapError::InvalidRopeBase {
            base_bits: base.to_bits(),
        });
    }
    Ok((0..layout.pairs)
        .map(|index| 1.0 / base.powf(2.0 * index as f32 / rope_dim as f32))
        .collect())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GpuNativeRopeLayout {
    rope_dim: usize,
    pairs: usize,
}

impl GpuNativeRopeLayout {
    fn try_new(rope_dim: usize, head_dim: usize) -> Result<Self, GpuNativeBootstrapError> {
        validate_rope_dimension(rope_dim, head_dim)?;
        Ok(Self {
            rope_dim,
            pairs: rope_dim / 2,
        })
    }
}

/// Checked request-local per-layer F32 KV capacity. Each physical K or V
/// buffer stores `[max_seq_len, kv_width]` for exactly one layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeKvLayout {
    num_layers: usize,
    max_seq_len: usize,
    kv_width: usize,
    layer_elements: usize,
    layer_bytes: u64,
    total_bytes: u64,
}

impl GpuNativeKvLayout {
    fn try_new(
        num_layers: usize,
        max_seq_len: usize,
        kv_width: usize,
        limits: &wgpu::Limits,
    ) -> Result<Self, GpuNativeBootstrapError> {
        if num_layers == 0 {
            return Err(GpuNativeBootstrapError::InvalidKvLayerCount);
        }
        if max_seq_len == 0 {
            return Err(GpuNativeBootstrapError::InvalidKvCapacity);
        }
        if kv_width == 0 {
            return Err(GpuNativeBootstrapError::InvalidKvWidth);
        }
        let capacity_error = || GpuNativeBootstrapError::KvCapacityOverflow {
            num_layers,
            max_seq_len,
            kv_width,
        };
        let layer_elements = max_seq_len
            .checked_mul(kv_width)
            .ok_or_else(capacity_error)?;
        if u32::try_from(max_seq_len).is_err()
            || u32::try_from(kv_width).is_err()
            || u32::try_from(layer_elements).is_err()
        {
            return Err(capacity_error());
        }
        let layer_bytes = layer_elements
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(capacity_error)?;
        let maximum_binding = u64::from(limits.max_storage_buffer_binding_size);
        if layer_bytes > limits.max_buffer_size || layer_bytes > maximum_binding {
            return Err(GpuNativeBootstrapError::KvBufferLimit {
                required: layer_bytes,
                max_buffer_size: limits.max_buffer_size,
                max_storage_binding_size: maximum_binding,
            });
        }
        let num_layers_u64 = u64::try_from(num_layers).map_err(|_| capacity_error())?;
        let total_bytes = layer_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_mul(num_layers_u64))
            .ok_or_else(capacity_error)?;
        Ok(Self {
            num_layers,
            max_seq_len,
            kv_width,
            layer_elements,
            layer_bytes,
            total_bytes,
        })
    }

    fn usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE
    }

    fn validate_layer(self, layer: usize) -> Result<(), GpuNativeBootstrapError> {
        if layer >= self.num_layers {
            return Err(GpuNativeBootstrapError::InvalidKvLayer {
                layer,
                num_layers: self.num_layers,
            });
        }
        Ok(())
    }

    fn validate_position(self, position: usize) -> Result<(), GpuNativeBootstrapError> {
        if position >= self.max_seq_len {
            return Err(GpuNativeBootstrapError::InvalidKvPosition {
                position,
                max_seq_len: self.max_seq_len,
            });
        }
        Ok(())
    }

    fn element_offset(
        self,
        layer: usize,
        position: usize,
    ) -> Result<usize, GpuNativeBootstrapError> {
        self.validate_layer(layer)?;
        self.validate_position(position)?;
        position
            .checked_mul(self.kv_width)
            .ok_or(GpuNativeBootstrapError::KvCapacityOverflow {
                num_layers: self.num_layers,
                max_seq_len: self.max_seq_len,
                kv_width: self.kv_width,
            })
    }

    pub(crate) const fn num_layers(self) -> usize {
        self.num_layers
    }

    pub(crate) const fn max_seq_len(self) -> usize {
        self.max_seq_len
    }

    pub(crate) const fn kv_width(self) -> usize {
        self.kv_width
    }

    pub(crate) const fn layer_bytes(self) -> u64 {
        self.layer_bytes
    }

    pub(crate) const fn total_bytes(self) -> u64 {
        self.total_bytes
    }
}

/// Stable model-scoped key used to retrieve a registered dense tensor.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GpuNativeDenseWeightKey(Arc<str>);

impl GpuNativeDenseWeightKey {
    pub(crate) fn try_new(key: impl Into<String>) -> Result<Self, GpuNativeBootstrapError> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(GpuNativeBootstrapError::InvalidDenseWeightKey);
        }
        Ok(Self(Arc::<str>::from(key)))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

/// Opaque, context-bound reference to one persistent model weight.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeDenseWeightHandle {
    context_id: u64,
    weight_id: u64,
    key: GpuNativeDenseWeightKey,
    layout: GpuNativeDenseWeightLayout,
}

impl GpuNativeDenseWeightHandle {
    pub(crate) fn key(&self) -> &GpuNativeDenseWeightKey {
        &self.key
    }

    pub(crate) const fn layout(&self) -> GpuNativeDenseWeightLayout {
        self.layout
    }
}

/// Narrow semantic wrapper for a persistent F32 `[1, width]` gain vector.
/// The underlying registry identity remains model-scoped and context-bound,
/// while callers cannot accidentally use this handle as a matrix operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeRmsNormHandle {
    dense: GpuNativeDenseWeightHandle,
    width: usize,
}

impl GpuNativeRmsNormHandle {
    fn from_dense(dense: GpuNativeDenseWeightHandle) -> Self {
        Self {
            width: dense.layout.cols,
            dense,
        }
    }

    pub(crate) const fn width(&self) -> usize {
        self.width
    }
}

/// Persistent model-scoped RoPE parameters stored in the existing dense F32
/// registry. `inv_freq` is `[rope_dim / 2]`; `attention_factor` is folded into
/// both sine and cosine, matching the CPU helper when a derived scaling table
/// is registered.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeRopeHandle {
    dense: GpuNativeDenseWeightHandle,
    rope_dim: usize,
    attention_factor_bits: u32,
}

impl GpuNativeRopeHandle {
    pub(crate) const fn rope_dim(&self) -> usize {
        self.rope_dim
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeAttentionNorm {
    handle: GpuNativeRmsNormHandle,
    epsilon_bits: u32,
}

impl GpuNativeAttentionNorm {
    pub(crate) fn try_new(
        handle: GpuNativeRmsNormHandle,
        epsilon: f32,
    ) -> Result<Self, GpuNativeBootstrapError> {
        validate_rms_norm_epsilon(epsilon)?;
        Ok(Self {
            handle,
            epsilon_bits: epsilon.to_bits(),
        })
    }

    fn epsilon(&self) -> f32 {
        f32::from_bits(self.epsilon_bits)
    }
}

/// Immutable context-bound handles and geometry for one Qwen-compatible
/// attention layer. The layer index is part of the plan identity so request-
/// local KV buffers cannot be selected independently at encode time.
/// Projection bias, asymmetric V, attention sinks, sliding-window execution,
/// and post-attention scaling are outside this foundation contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeAttentionPlan {
    context_id: u64,
    layer_index: usize,
    geometry: GpuNativeAttentionGeometry,
    q_projection: GpuNativeDenseWeightHandle,
    k_projection: GpuNativeDenseWeightHandle,
    v_projection: GpuNativeDenseWeightHandle,
    o_projection: GpuNativeDenseWeightHandle,
    q_norm: Option<GpuNativeAttentionNorm>,
    k_norm: Option<GpuNativeAttentionNorm>,
    rope: GpuNativeRopeHandle,
}

impl GpuNativeAttentionPlan {
    pub(crate) const fn geometry(&self) -> GpuNativeAttentionGeometry {
        self.geometry
    }

    pub(crate) const fn layer_index(&self) -> usize {
        self.layer_index
    }
}

struct GpuNativeDenseWeightChunk<B = wgpu::Buffer> {
    plan: GpuNativeDenseWeightChunkPlan,
    buffer: B,
}

struct GpuNativeDenseWeight<B = wgpu::Buffer> {
    weight_id: u64,
    key: GpuNativeDenseWeightKey,
    layout: GpuNativeDenseWeightLayout,
    chunks: Vec<GpuNativeDenseWeightChunk<B>>,
}

impl<B> GpuNativeDenseWeight<B> {
    fn handle(&self, context_id: u64) -> GpuNativeDenseWeightHandle {
        GpuNativeDenseWeightHandle {
            context_id,
            weight_id: self.weight_id,
            key: self.key.clone(),
            layout: self.layout,
        }
    }
}

struct GpuNativeDenseWeightRegistry<B = wgpu::Buffer> {
    context_id: u64,
    weights: HashMap<GpuNativeDenseWeightKey, Arc<GpuNativeDenseWeight<B>>>,
}

impl<B> GpuNativeDenseWeightRegistry<B> {
    fn new(context_id: u64) -> Self {
        Self {
            context_id,
            weights: HashMap::new(),
        }
    }

    fn insert(
        &mut self,
        weight: GpuNativeDenseWeight<B>,
    ) -> Result<GpuNativeDenseWeightHandle, GpuNativeBootstrapError> {
        if self.weights.contains_key(&weight.key) {
            return Err(GpuNativeBootstrapError::DuplicateDenseWeight {
                key: weight.key.as_str().to_string(),
            });
        }
        let weight = Arc::new(weight);
        let handle = weight.handle(self.context_id);
        self.weights.insert(weight.key.clone(), weight);
        Ok(handle)
    }

    fn resolve(
        &self,
        handle: &GpuNativeDenseWeightHandle,
    ) -> Result<Arc<GpuNativeDenseWeight<B>>, GpuNativeBootstrapError> {
        if handle.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignDenseWeightHandle);
        }
        let weight = self.weights.get(&handle.key).ok_or_else(|| {
            GpuNativeBootstrapError::MissingDenseWeight {
                key: handle.key.as_str().to_string(),
            }
        })?;
        if weight.weight_id != handle.weight_id || weight.layout != handle.layout {
            return Err(GpuNativeBootstrapError::StaleDenseWeightHandle {
                key: handle.key.as_str().to_string(),
            });
        }
        Ok(weight.clone())
    }

    fn handle_for(
        &self,
        key: &GpuNativeDenseWeightKey,
        expected_kind: GpuNativeDenseWeightKind,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<GpuNativeDenseWeightHandle, GpuNativeBootstrapError> {
        let weight =
            self.weights
                .get(key)
                .ok_or_else(|| GpuNativeBootstrapError::MissingDenseWeight {
                    key: key.as_str().to_string(),
                })?;
        if weight.layout.kind != expected_kind {
            return Err(GpuNativeBootstrapError::DenseWeightKindMismatch {
                key: key.as_str().to_string(),
                expected: expected_kind,
                actual: weight.layout.kind,
            });
        }
        if weight.layout.rows != expected_rows || weight.layout.cols != expected_cols {
            return Err(GpuNativeBootstrapError::DenseWeightShapeMismatch {
                key: key.as_str().to_string(),
                expected_rows,
                expected_cols,
                actual_rows: weight.layout.rows,
                actual_cols: weight.layout.cols,
            });
        }
        Ok(weight.handle(self.context_id))
    }

    fn resolve_rms_norm(
        &self,
        handle: &GpuNativeRmsNormHandle,
    ) -> Result<Arc<GpuNativeDenseWeight<B>>, GpuNativeBootstrapError> {
        if handle.dense.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignRmsNormHandle);
        }
        let weight = self.weights.get(&handle.dense.key).ok_or_else(|| {
            GpuNativeBootstrapError::MissingDenseWeight {
                key: handle.dense.key.as_str().to_string(),
            }
        })?;
        if weight.weight_id != handle.dense.weight_id
            || weight.layout != handle.dense.layout
            || handle.width != handle.dense.layout.cols
        {
            return Err(GpuNativeBootstrapError::StaleRmsNormHandle {
                key: handle.dense.key.as_str().to_string(),
            });
        }
        if weight.layout.kind != GpuNativeDenseWeightKind::F32
            || weight.layout.rows != 1
            || weight.layout.cols != handle.width
        {
            return Err(GpuNativeBootstrapError::StaleRmsNormHandle {
                key: handle.dense.key.as_str().to_string(),
            });
        }
        Ok(weight.clone())
    }
}

fn validate_rope_handle_with_registry<B>(
    context_id: u64,
    registry: &GpuNativeDenseWeightRegistry<B>,
    handle: &GpuNativeRopeHandle,
    expected_rope_dim: usize,
) -> Result<(), GpuNativeBootstrapError> {
    if handle.dense.context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignRopeHandle);
    }
    if handle.rope_dim != expected_rope_dim {
        return Err(GpuNativeBootstrapError::RopeDimensionMismatch {
            expected: expected_rope_dim,
            actual: handle.rope_dim,
        });
    }
    let weight = registry
        .resolve(&handle.dense)
        .map_err(|error| match error {
            GpuNativeBootstrapError::ForeignDenseWeightHandle => {
                GpuNativeBootstrapError::ForeignRopeHandle
            }
            _ => GpuNativeBootstrapError::StaleRopeHandle {
                key: handle.dense.key.as_str().to_string(),
            },
        })?;
    if weight.layout.kind != GpuNativeDenseWeightKind::F32
        || weight.layout.rows != 1
        || weight.layout.cols != handle.rope_dim / 2
        || !f32::from_bits(handle.attention_factor_bits).is_finite()
        || f32::from_bits(handle.attention_factor_bits) <= 0.0
    {
        return Err(GpuNativeBootstrapError::StaleRopeHandle {
            key: handle.dense.key.as_str().to_string(),
        });
    }
    Ok(())
}

static NEXT_GPU_NATIVE_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_GPU_NATIVE_WEIGHT_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_GPU_NATIVE_SCRATCH_ID: AtomicU64 = AtomicU64::new(1);

fn next_nonzero_id(counter: &AtomicU64, label: &str) -> u64 {
    let id = counter.fetch_add(1, Ordering::Relaxed);
    assert_ne!(id, 0, "GPU-native {label} id space exhausted");
    id
}

/// Checked byte layout for one request's initial device-resident token state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeTokenStateLayout {
    d_model: usize,
    vector_bytes: u64,
    status_bytes: u64,
    total_buffer_bytes: u64,
}

impl GpuNativeTokenStateLayout {
    pub(crate) fn try_new(d_model: usize) -> Result<Self, GpuNativeBootstrapError> {
        if d_model == 0 {
            return Err(GpuNativeBootstrapError::InvalidDModel);
        }

        let vector_bytes = d_model
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(GpuNativeBootstrapError::StateSizeOverflow { d_model })?;
        let total_buffer_bytes = vector_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(GPU_NATIVE_STATUS_BYTES))
            .ok_or(GpuNativeBootstrapError::StateSizeOverflow { d_model })?;

        Ok(Self {
            d_model,
            vector_bytes,
            status_bytes: GPU_NATIVE_STATUS_BYTES,
            total_buffer_bytes,
        })
    }

    pub(crate) const fn d_model(self) -> usize {
        self.d_model
    }

    pub(crate) const fn vector_bytes(self) -> u64 {
        self.vector_bytes
    }

    pub(crate) const fn status_bytes(self) -> u64 {
        self.status_bytes
    }

    pub(crate) const fn total_buffer_bytes(self) -> u64 {
        self.total_buffer_bytes
    }

    fn tensor_usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
    }

    fn status_usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
    }

    fn validate_for_limits(self, limits: &wgpu::Limits) -> Result<(), GpuNativeBootstrapError> {
        super::validate_startup_buffer(
            "gpu_native_hidden",
            self.vector_bytes,
            Self::tensor_usage(),
            limits,
        )?;
        super::validate_startup_buffer(
            "gpu_native_residual",
            self.vector_bytes,
            Self::tensor_usage(),
            limits,
        )?;
        super::validate_startup_buffer(
            "gpu_native_status",
            self.status_bytes,
            Self::status_usage(),
            limits,
        )?;
        Ok(())
    }
}

/// Checked request-scoped F32 device scratch for variable-width GEMV output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeScratchLayout {
    elements: usize,
    bytes: u64,
}

impl GpuNativeScratchLayout {
    pub(crate) fn try_new(elements: usize) -> Result<Self, GpuNativeBootstrapError> {
        if elements == 0 {
            return Err(GpuNativeBootstrapError::InvalidScratchElements);
        }
        if u32::try_from(elements).is_err() {
            return Err(GpuNativeBootstrapError::ScratchSizeOverflow { elements });
        }
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(GpuNativeBootstrapError::ScratchSizeOverflow { elements })?;
        Ok(Self { elements, bytes })
    }

    pub(crate) const fn elements(self) -> usize {
        self.elements
    }

    pub(crate) const fn bytes(self) -> u64 {
        self.bytes
    }

    fn usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
    }

    fn validate_for_limits(self, limits: &wgpu::Limits) -> Result<(), GpuNativeBootstrapError> {
        super::validate_startup_buffer("gpu_native_scratch", self.bytes, Self::usage(), limits)?;
        Ok(())
    }
}

pub(crate) struct GpuNativeScratch<B = wgpu::Buffer> {
    context_id: u64,
    scratch_id: u64,
    layout: GpuNativeScratchLayout,
    buffer: B,
}

impl<B> GpuNativeScratch<B> {
    fn from_buffer(
        context_id: u64,
        scratch_id: u64,
        layout: GpuNativeScratchLayout,
        buffer: B,
    ) -> Self {
        Self {
            context_id,
            scratch_id,
            layout,
            buffer,
        }
    }

    pub(crate) const fn layout(&self) -> GpuNativeScratchLayout {
        self.layout
    }
}

impl<B> fmt::Debug for GpuNativeScratch<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuNativeScratch")
            .field("scratch_id", &self.scratch_id)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

/// Request-scoped, non-mappable F32 attention intermediates with geometry
/// attached, preventing projection, context, and output buffers from being
/// interchanged at the composed API.
pub(crate) struct GpuNativeAttentionScratch<B = wgpu::Buffer> {
    context_id: u64,
    geometry: GpuNativeAttentionGeometry,
    q: GpuNativeScratch<B>,
    k: GpuNativeScratch<B>,
    v: GpuNativeScratch<B>,
    context: GpuNativeScratch<B>,
    projected: GpuNativeScratch<B>,
}

impl<B> GpuNativeAttentionScratch<B> {
    fn from_scratch(
        context_id: u64,
        geometry: GpuNativeAttentionGeometry,
        q: GpuNativeScratch<B>,
        k: GpuNativeScratch<B>,
        v: GpuNativeScratch<B>,
        context: GpuNativeScratch<B>,
        projected: GpuNativeScratch<B>,
    ) -> Self {
        Self {
            context_id,
            geometry,
            q,
            k,
            v,
            context,
            projected,
        }
    }

    pub(crate) const fn geometry(&self) -> GpuNativeAttentionGeometry {
        self.geometry
    }
}

impl<B> fmt::Debug for GpuNativeAttentionScratch<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuNativeAttentionScratch")
            .field("geometry", &self.geometry)
            .field("q", &self.q)
            .field("k", &self.k)
            .field("v", &self.v)
            .field("context", &self.context)
            .field("projected", &self.projected)
            .finish()
    }
}

struct GpuNativeKvLayer<B = wgpu::Buffer> {
    key: B,
    value: B,
}

/// Explicitly request-local F32 KV storage. Physical buffers are per-layer so
/// no single WGPU storage binding grows with the model's layer count.
pub(crate) struct GpuNativeKvState<B = wgpu::Buffer> {
    context_id: u64,
    kv_id: u64,
    layout: GpuNativeKvLayout,
    layers: Vec<GpuNativeKvLayer<B>>,
}

impl<B> GpuNativeKvState<B> {
    fn from_layers(
        context_id: u64,
        kv_id: u64,
        layout: GpuNativeKvLayout,
        layers: Vec<GpuNativeKvLayer<B>>,
    ) -> Self {
        Self {
            context_id,
            kv_id,
            layout,
            layers,
        }
    }

    pub(crate) const fn layout(&self) -> GpuNativeKvLayout {
        self.layout
    }
}

impl<B> fmt::Debug for GpuNativeKvState<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuNativeKvState")
            .field("kv_id", &self.kv_id)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

static NEXT_GPU_NATIVE_KV_ID: AtomicU64 = AtomicU64::new(1);

static NEXT_GPU_NATIVE_TOKEN_STATE_ID: AtomicU64 = AtomicU64::new(1);

fn next_gpu_native_token_state_id() -> u64 {
    let id = NEXT_GPU_NATIVE_TOKEN_STATE_ID.fetch_add(1, Ordering::Relaxed);
    assert_ne!(id, 0, "GPU-native token state id space exhausted");
    id
}

/// Opaque request-local ownership of mutable GPU-native token buffers.
///
/// The generic defaults to the production WGPU buffer type. The parameter
/// lets hardware-independent tests exercise ownership and drop behavior
/// without introducing a mock WGPU device.
pub(crate) struct GpuNativeTokenState<B = wgpu::Buffer> {
    context_id: u64,
    state_id: u64,
    layout: GpuNativeTokenStateLayout,
    hidden: B,
    residual: B,
    status: B,
}

impl<B> GpuNativeTokenState<B> {
    fn from_buffers(
        context_id: u64,
        state_id: u64,
        layout: GpuNativeTokenStateLayout,
        hidden: B,
        residual: B,
        status: B,
    ) -> Self {
        Self {
            context_id,
            state_id,
            layout,
            hidden,
            residual,
            status,
        }
    }

    pub(crate) const fn state_id(&self) -> u64 {
        self.state_id
    }

    pub(crate) const fn layout(&self) -> GpuNativeTokenStateLayout {
        self.layout
    }
}

impl<B> fmt::Debug for GpuNativeTokenState<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuNativeTokenState")
            .field("state_id", &self.state_id)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

fn validate_token_state_owner(
    context_id: u64,
    state_context_id: u64,
) -> Result<(), GpuNativeBootstrapError> {
    if state_context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignTokenState);
    }
    Ok(())
}

fn validate_scratch_owner(
    context_id: u64,
    scratch_context_id: u64,
) -> Result<(), GpuNativeBootstrapError> {
    if scratch_context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignScratch);
    }
    Ok(())
}

fn validate_residual_contribution_width(
    expected: usize,
    actual: usize,
) -> Result<(), GpuNativeBootstrapError> {
    if actual != expected {
        return Err(GpuNativeBootstrapError::ResidualContributionWidth { expected, actual });
    }
    Ok(())
}

fn validate_rms_norm_weight_width(width: usize) -> Result<(), GpuNativeBootstrapError> {
    if width == 0 {
        return Err(GpuNativeBootstrapError::InvalidRmsNormWeightWidth { width });
    }
    Ok(())
}

fn validate_rms_norm_epsilon(epsilon: f32) -> Result<(), GpuNativeBootstrapError> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(GpuNativeBootstrapError::InvalidRmsNormEpsilon {
            epsilon_bits: epsilon.to_bits(),
        });
    }
    Ok(())
}

fn validate_attention_scratch<B>(
    context_id: u64,
    geometry: GpuNativeAttentionGeometry,
    scratch: &GpuNativeAttentionScratch<B>,
) -> Result<(), GpuNativeBootstrapError> {
    if scratch.context_id != context_id
        || scratch.q.context_id != context_id
        || scratch.k.context_id != context_id
        || scratch.v.context_id != context_id
        || scratch.context.context_id != context_id
        || scratch.projected.context_id != context_id
    {
        return Err(GpuNativeBootstrapError::ForeignAttentionScratch);
    }
    let scratch_ids = [
        scratch.q.scratch_id,
        scratch.k.scratch_id,
        scratch.v.scratch_id,
        scratch.context.scratch_id,
        scratch.projected.scratch_id,
    ];
    for (index, scratch_id) in scratch_ids.iter().copied().enumerate() {
        if scratch_ids[index + 1..].contains(&scratch_id) {
            return Err(GpuNativeBootstrapError::AliasedInputOutput);
        }
    }
    for (tensor, expected, actual) in [
        (
            GpuNativeAttentionTensor::Query,
            geometry.q_width,
            scratch.q.layout.elements,
        ),
        (
            GpuNativeAttentionTensor::Key,
            geometry.kv_width,
            scratch.k.layout.elements,
        ),
        (
            GpuNativeAttentionTensor::Value,
            geometry.kv_width,
            scratch.v.layout.elements,
        ),
        (
            GpuNativeAttentionTensor::Context,
            geometry.q_width,
            scratch.context.layout.elements,
        ),
        (
            GpuNativeAttentionTensor::Output,
            geometry.d_model,
            scratch.projected.layout.elements,
        ),
    ] {
        if actual != expected {
            return Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor,
                expected,
                actual,
            });
        }
    }
    if scratch.geometry != geometry {
        return Err(GpuNativeBootstrapError::AttentionScratchGeometry {
            expected: geometry,
            actual: scratch.geometry,
        });
    }
    Ok(())
}

fn validate_attention_plan_with_registry<B>(
    context_id: u64,
    d_model: usize,
    registry: &GpuNativeDenseWeightRegistry<B>,
    plan: &GpuNativeAttentionPlan,
) -> Result<(), GpuNativeBootstrapError> {
    if plan.context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignAttentionPlan);
    }
    if plan.geometry.d_model != d_model {
        return Err(GpuNativeBootstrapError::AttentionDModelMismatch {
            expected: d_model,
            actual: plan.geometry.d_model,
        });
    }
    for (tensor, handle, rows, cols) in [
        (
            GpuNativeAttentionTensor::Query,
            &plan.q_projection,
            plan.geometry.q_width,
            plan.geometry.d_model,
        ),
        (
            GpuNativeAttentionTensor::Key,
            &plan.k_projection,
            plan.geometry.kv_width,
            plan.geometry.d_model,
        ),
        (
            GpuNativeAttentionTensor::Value,
            &plan.v_projection,
            plan.geometry.kv_width,
            plan.geometry.d_model,
        ),
        (
            GpuNativeAttentionTensor::Output,
            &plan.o_projection,
            plan.geometry.d_model,
            plan.geometry.q_width,
        ),
    ] {
        let weight = registry.resolve(handle)?;
        if weight.layout.rows != rows || weight.layout.cols != cols {
            return Err(GpuNativeBootstrapError::AttentionProjectionShape {
                tensor,
                expected_rows: rows,
                expected_cols: cols,
                actual_rows: weight.layout.rows,
                actual_cols: weight.layout.cols,
            });
        }
    }
    for (tensor, norm) in [
        (GpuNativeAttentionTensor::Query, plan.q_norm.as_ref()),
        (GpuNativeAttentionTensor::Key, plan.k_norm.as_ref()),
    ] {
        let Some(norm) = norm else {
            continue;
        };
        validate_rms_norm_epsilon(norm.epsilon())?;
        let weight = registry.resolve_rms_norm(&norm.handle)?;
        if weight.layout.cols != plan.geometry.head_dim {
            return Err(GpuNativeBootstrapError::AttentionNormWidth {
                tensor,
                expected: plan.geometry.head_dim,
                actual: weight.layout.cols,
            });
        }
    }
    validate_rope_handle_with_registry(context_id, registry, &plan.rope, plan.geometry.rope_dim)?;
    Ok(())
}

fn validate_kv_state<B>(
    context_id: u64,
    expected_width: usize,
    kv: &GpuNativeKvState<B>,
    layer: usize,
    position: usize,
) -> Result<(), GpuNativeBootstrapError> {
    if kv.context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignKvState);
    }
    if kv.layout.kv_width != expected_width {
        return Err(GpuNativeBootstrapError::KvWidth {
            expected: expected_width,
            actual: kv.layout.kv_width,
        });
    }
    kv.layout.validate_layer(layer)?;
    kv.layout.validate_position(position)?;
    Ok(())
}

fn validate_attention_kv_state<B>(
    context_id: u64,
    geometry: GpuNativeAttentionGeometry,
    kv: &GpuNativeKvState<B>,
    layer_index: usize,
    position: usize,
) -> Result<usize, GpuNativeBootstrapError> {
    if kv.context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignKvState);
    }
    if kv.layout.kv_width != geometry.kv_width {
        return Err(GpuNativeBootstrapError::KvWidth {
            expected: geometry.kv_width,
            actual: kv.layout.kv_width,
        });
    }
    if layer_index >= kv.layout.num_layers {
        return Err(GpuNativeBootstrapError::AttentionPlanLayerOutOfRange {
            layer_index,
            num_layers: kv.layout.num_layers,
        });
    }
    let seq_len = position
        .checked_add(1)
        .ok_or(GpuNativeBootstrapError::AttentionSequenceLengthOverflow { position })?;
    if seq_len == 0 || seq_len > kv.layout.max_seq_len {
        return Err(GpuNativeBootstrapError::InvalidAttentionSequenceLength {
            seq_len,
            max_seq_len: kv.layout.max_seq_len,
        });
    }
    Ok(seq_len)
}

/// Immutable, serializable evidence for GPU-native execution boundaries.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct GpuNativeExecutionSnapshot {
    pub(crate) dense_weights_registered: u64,
    pub(crate) dense_weight_chunks: u64,
    pub(crate) dense_weight_uploads: u64,
    pub(crate) dense_weight_upload_bytes: u64,
    pub(crate) dense_weight_resident_bytes: u64,
    pub(crate) dense_gemv_dispatches: u64,
    pub(crate) dense_gemv_chunk_dispatches: u64,
    pub(crate) embedding_dispatches: u64,
    pub(crate) rms_norm_dispatches: u64,
    pub(crate) rms_norm_groups: u64,
    pub(crate) rms_norm_state_dispatches: u64,
    pub(crate) rms_norm_scratch_dispatches: u64,
    pub(crate) residual_add_dispatches: u64,
    pub(crate) rope_parameters_registered: u64,
    pub(crate) rope_parameter_uploads: u64,
    pub(crate) rope_parameter_upload_bytes: u64,
    pub(crate) attention_prepare_dispatches: u64,
    pub(crate) q_projection_dispatches: u64,
    pub(crate) k_projection_dispatches: u64,
    pub(crate) v_projection_dispatches: u64,
    pub(crate) rope_dispatches: u64,
    pub(crate) rope_groups: u64,
    pub(crate) kv_appends: u64,
    pub(crate) causal_attention_dispatches: u64,
    pub(crate) o_projection_dispatches: u64,
    pub(crate) attention_complete_dispatches: u64,
    pub(crate) tokens_submitted: u64,
    pub(crate) tokens_completed: u64,
    pub(crate) layers_encoded: u64,
    pub(crate) queue_submissions: u64,
    pub(crate) token_boundary_maps: u64,
    pub(crate) token_boundary_readbacks: u64,
    pub(crate) intermediate_maps: u64,
    pub(crate) intermediate_readbacks: u64,
    pub(crate) cpu_layer_reentries: u64,
    pub(crate) cpu_attention_calls: u64,
    pub(crate) cpu_kv_mutations: u64,
    pub(crate) cpu_router_calls: u64,
    pub(crate) cpu_expert_combines: u64,
    pub(crate) expert_slot_misses: u64,
    pub(crate) device_loss_failures: u64,
    pub(crate) numerical_failures: u64,
}

#[derive(Debug, Default)]
struct GpuNativeExecutionCounters {
    dense_weights_registered: AtomicU64,
    dense_weight_chunks: AtomicU64,
    dense_weight_uploads: AtomicU64,
    dense_weight_upload_bytes: AtomicU64,
    dense_weight_resident_bytes: AtomicU64,
    dense_gemv_dispatches: AtomicU64,
    dense_gemv_chunk_dispatches: AtomicU64,
    embedding_dispatches: AtomicU64,
    rms_norm_dispatches: AtomicU64,
    rms_norm_groups: AtomicU64,
    rms_norm_state_dispatches: AtomicU64,
    rms_norm_scratch_dispatches: AtomicU64,
    residual_add_dispatches: AtomicU64,
    rope_parameters_registered: AtomicU64,
    rope_parameter_uploads: AtomicU64,
    rope_parameter_upload_bytes: AtomicU64,
    attention_prepare_dispatches: AtomicU64,
    q_projection_dispatches: AtomicU64,
    k_projection_dispatches: AtomicU64,
    v_projection_dispatches: AtomicU64,
    rope_dispatches: AtomicU64,
    rope_groups: AtomicU64,
    kv_appends: AtomicU64,
    causal_attention_dispatches: AtomicU64,
    o_projection_dispatches: AtomicU64,
    attention_complete_dispatches: AtomicU64,
    tokens_submitted: AtomicU64,
    tokens_completed: AtomicU64,
    layers_encoded: AtomicU64,
    queue_submissions: AtomicU64,
    token_boundary_maps: AtomicU64,
    token_boundary_readbacks: AtomicU64,
    intermediate_maps: AtomicU64,
    intermediate_readbacks: AtomicU64,
    cpu_layer_reentries: AtomicU64,
    cpu_attention_calls: AtomicU64,
    cpu_kv_mutations: AtomicU64,
    cpu_router_calls: AtomicU64,
    cpu_expert_combines: AtomicU64,
    expert_slot_misses: AtomicU64,
    device_loss_failures: AtomicU64,
    numerical_failures: AtomicU64,
}

impl GpuNativeExecutionCounters {
    fn snapshot(&self) -> GpuNativeExecutionSnapshot {
        GpuNativeExecutionSnapshot {
            dense_weights_registered: self.dense_weights_registered.load(Ordering::Relaxed),
            dense_weight_chunks: self.dense_weight_chunks.load(Ordering::Relaxed),
            dense_weight_uploads: self.dense_weight_uploads.load(Ordering::Relaxed),
            dense_weight_upload_bytes: self.dense_weight_upload_bytes.load(Ordering::Relaxed),
            dense_weight_resident_bytes: self.dense_weight_resident_bytes.load(Ordering::Relaxed),
            dense_gemv_dispatches: self.dense_gemv_dispatches.load(Ordering::Relaxed),
            dense_gemv_chunk_dispatches: self.dense_gemv_chunk_dispatches.load(Ordering::Relaxed),
            embedding_dispatches: self.embedding_dispatches.load(Ordering::Relaxed),
            rms_norm_dispatches: self.rms_norm_dispatches.load(Ordering::Relaxed),
            rms_norm_groups: self.rms_norm_groups.load(Ordering::Relaxed),
            rms_norm_state_dispatches: self.rms_norm_state_dispatches.load(Ordering::Relaxed),
            rms_norm_scratch_dispatches: self.rms_norm_scratch_dispatches.load(Ordering::Relaxed),
            residual_add_dispatches: self.residual_add_dispatches.load(Ordering::Relaxed),
            rope_parameters_registered: self.rope_parameters_registered.load(Ordering::Relaxed),
            rope_parameter_uploads: self.rope_parameter_uploads.load(Ordering::Relaxed),
            rope_parameter_upload_bytes: self.rope_parameter_upload_bytes.load(Ordering::Relaxed),
            attention_prepare_dispatches: self.attention_prepare_dispatches.load(Ordering::Relaxed),
            q_projection_dispatches: self.q_projection_dispatches.load(Ordering::Relaxed),
            k_projection_dispatches: self.k_projection_dispatches.load(Ordering::Relaxed),
            v_projection_dispatches: self.v_projection_dispatches.load(Ordering::Relaxed),
            rope_dispatches: self.rope_dispatches.load(Ordering::Relaxed),
            rope_groups: self.rope_groups.load(Ordering::Relaxed),
            kv_appends: self.kv_appends.load(Ordering::Relaxed),
            causal_attention_dispatches: self.causal_attention_dispatches.load(Ordering::Relaxed),
            o_projection_dispatches: self.o_projection_dispatches.load(Ordering::Relaxed),
            attention_complete_dispatches: self
                .attention_complete_dispatches
                .load(Ordering::Relaxed),
            tokens_submitted: self.tokens_submitted.load(Ordering::Relaxed),
            tokens_completed: self.tokens_completed.load(Ordering::Relaxed),
            layers_encoded: self.layers_encoded.load(Ordering::Relaxed),
            queue_submissions: self.queue_submissions.load(Ordering::Relaxed),
            token_boundary_maps: self.token_boundary_maps.load(Ordering::Relaxed),
            token_boundary_readbacks: self.token_boundary_readbacks.load(Ordering::Relaxed),
            intermediate_maps: self.intermediate_maps.load(Ordering::Relaxed),
            intermediate_readbacks: self.intermediate_readbacks.load(Ordering::Relaxed),
            cpu_layer_reentries: self.cpu_layer_reentries.load(Ordering::Relaxed),
            cpu_attention_calls: self.cpu_attention_calls.load(Ordering::Relaxed),
            cpu_kv_mutations: self.cpu_kv_mutations.load(Ordering::Relaxed),
            cpu_router_calls: self.cpu_router_calls.load(Ordering::Relaxed),
            cpu_expert_combines: self.cpu_expert_combines.load(Ordering::Relaxed),
            expert_slot_misses: self.expert_slot_misses.load(Ordering::Relaxed),
            device_loss_failures: self.device_loss_failures.load(Ordering::Relaxed),
            numerical_failures: self.numerical_failures.load(Ordering::Relaxed),
        }
    }

    fn record_dense_weight_registration(&self, chunks: u64, allocation_bytes: u64) {
        self.dense_weights_registered
            .fetch_add(1, Ordering::Relaxed);
        self.dense_weight_chunks
            .fetch_add(chunks, Ordering::Relaxed);
        self.dense_weight_uploads
            .fetch_add(chunks, Ordering::Relaxed);
        self.dense_weight_upload_bytes
            .fetch_add(allocation_bytes, Ordering::Relaxed);
        self.dense_weight_resident_bytes
            .fetch_add(allocation_bytes, Ordering::Relaxed);
    }

    fn record_dense_gemv_dispatch(&self, chunks: u64) {
        self.dense_gemv_dispatches.fetch_add(1, Ordering::Relaxed);
        self.dense_gemv_chunk_dispatches
            .fetch_add(chunks, Ordering::Relaxed);
    }

    fn record_embedding_dispatch(&self) {
        self.embedding_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    fn record_rms_norm_state_dispatch(&self, groups: u64) {
        self.rms_norm_dispatches.fetch_add(1, Ordering::Relaxed);
        self.rms_norm_groups.fetch_add(groups, Ordering::Relaxed);
        self.rms_norm_state_dispatches
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_rms_norm_scratch_dispatch(&self, groups: u64) {
        self.rms_norm_dispatches.fetch_add(1, Ordering::Relaxed);
        self.rms_norm_groups.fetch_add(groups, Ordering::Relaxed);
        self.rms_norm_scratch_dispatches
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_residual_add_dispatch(&self) {
        self.residual_add_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    fn record_rope_registration(&self, upload_bytes: u64) {
        self.rope_parameters_registered
            .fetch_add(1, Ordering::Relaxed);
        self.rope_parameter_uploads.fetch_add(1, Ordering::Relaxed);
        self.rope_parameter_upload_bytes
            .fetch_add(upload_bytes, Ordering::Relaxed);
    }

    fn record_attention_prepare_dispatch(&self) {
        self.attention_prepare_dispatches
            .fetch_add(1, Ordering::Relaxed);
        self.q_projection_dispatches.fetch_add(1, Ordering::Relaxed);
        self.k_projection_dispatches.fetch_add(1, Ordering::Relaxed);
        self.v_projection_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    fn record_rope_dispatch(&self, groups: u64) {
        self.rope_dispatches.fetch_add(1, Ordering::Relaxed);
        self.rope_groups.fetch_add(groups, Ordering::Relaxed);
    }

    fn record_kv_append(&self) {
        self.kv_appends.fetch_add(1, Ordering::Relaxed);
    }

    fn record_causal_attention_dispatch(&self) {
        self.causal_attention_dispatches
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_attention_complete_dispatch(&self) {
        self.o_projection_dispatches.fetch_add(1, Ordering::Relaxed);
        self.attention_complete_dispatches
            .fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn record_token_boundary_readback(&self) {
        self.token_boundary_readbacks
            .fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn record_intermediate_readback(&self) {
        self.intermediate_readbacks.fetch_add(1, Ordering::Relaxed);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeGemvPushConstants {
    rows: u32,
    cols: u32,
    global_row_base: u32,
    q8_first_block: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeEmbeddingPushConstants {
    local_row: u32,
    global_row: u32,
    cols: u32,
    q8_first_block: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeRmsNormPushConstants {
    groups: u32,
    group_width: u32,
    epsilon_bits: u32,
    _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeRopePushConstants {
    groups: u32,
    head_dim: u32,
    rope_dim: u32,
    position: u32,
    attention_factor_bits: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeKvAppendPushConstants {
    width: u32,
    position: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeAttentionPushConstants {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
}

struct GpuNativeDensePipelines {
    gemv_bind_group_layout: wgpu::BindGroupLayout,
    embedding_bind_group_layout: wgpu::BindGroupLayout,
    f32_gemv: wgpu::ComputePipeline,
    q8_0_gemv: wgpu::ComputePipeline,
    f32_embedding: wgpu::ComputePipeline,
    q8_0_embedding: wgpu::ComputePipeline,
}

impl GpuNativeDensePipelines {
    fn new(device: &wgpu::Device) -> Self {
        let read_only_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let read_write_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let gemv_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_dense_gemv_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                ],
            });
        let embedding_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_embedding_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                ],
            });
        let gemv_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_native_dense_gemv_pipeline_layout"),
            bind_group_layouts: &[&gemv_bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });
        let embedding_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_native_embedding_pipeline_layout"),
                bind_group_layouts: &[&embedding_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let gemv_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_dense_gemv_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_DENSE_GEMV_SHADER.into()),
        });
        let embedding_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_embedding_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_EMBEDDING_SHADER.into()),
        });
        let pipeline = |label: &'static str,
                        layout: &wgpu::PipelineLayout,
                        module: &wgpu::ShaderModule,
                        entry_point: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module,
                entry_point,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        };
        let f32_gemv = pipeline(
            "gpu_native_f32_gemv_pipeline",
            &gemv_pipeline_layout,
            &gemv_module,
            "f32_gemv_main",
        );
        let q8_0_gemv = pipeline(
            "gpu_native_q8_0_gemv_pipeline",
            &gemv_pipeline_layout,
            &gemv_module,
            "q8_0_gemv_main",
        );
        let f32_embedding = pipeline(
            "gpu_native_f32_embedding_pipeline",
            &embedding_pipeline_layout,
            &embedding_module,
            "f32_embedding_main",
        );
        let q8_0_embedding = pipeline(
            "gpu_native_q8_0_embedding_pipeline",
            &embedding_pipeline_layout,
            &embedding_module,
            "q8_0_embedding_main",
        );
        Self {
            gemv_bind_group_layout,
            embedding_bind_group_layout,
            f32_gemv,
            q8_0_gemv,
            f32_embedding,
            q8_0_embedding,
        }
    }
}

struct GpuNativeStatePipelines {
    rms_capture_bind_group_layout: wgpu::BindGroupLayout,
    rms_in_place_bind_group_layout: wgpu::BindGroupLayout,
    rms_capture: wgpu::ComputePipeline,
    rms_in_place: wgpu::ComputePipeline,
    residual_add: wgpu::ComputePipeline,
}

impl GpuNativeStatePipelines {
    fn new(device: &wgpu::Device) -> Self {
        let read_only_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let read_write_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let rms_capture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_rms_capture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                ],
            });
        let rms_in_place_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_rms_in_place_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                ],
            });
        let pipeline_layout = |label, bind_group_layout: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            })
        };
        let capture_pipeline_layout = pipeline_layout(
            "gpu_native_rms_capture_pipeline_layout",
            &rms_capture_bind_group_layout,
        );
        let in_place_pipeline_layout = pipeline_layout(
            "gpu_native_rms_in_place_pipeline_layout",
            &rms_in_place_bind_group_layout,
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_rmsnorm_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_RMSNORM_SHADER.into()),
        });
        let pipeline =
            |label: &'static str, layout: &wgpu::PipelineLayout, entry_point: &'static str| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    module: &module,
                    entry_point,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                })
            };
        let rms_capture = pipeline(
            "gpu_native_rms_capture_pipeline",
            &capture_pipeline_layout,
            "rms_norm_capture_main",
        );
        let rms_in_place = pipeline(
            "gpu_native_rms_in_place_pipeline",
            &in_place_pipeline_layout,
            "rms_norm_in_place_main",
        );
        let residual_add = pipeline(
            "gpu_native_residual_add_pipeline",
            &capture_pipeline_layout,
            "residual_add_main",
        );
        Self {
            rms_capture_bind_group_layout,
            rms_in_place_bind_group_layout,
            rms_capture,
            rms_in_place,
            residual_add,
        }
    }
}

struct GpuNativeAttentionPipelines {
    rope_bind_group_layout: wgpu::BindGroupLayout,
    kv_append_bind_group_layout: wgpu::BindGroupLayout,
    causal_attention_bind_group_layout: wgpu::BindGroupLayout,
    rope: wgpu::ComputePipeline,
    kv_append: wgpu::ComputePipeline,
    causal_attention: wgpu::ComputePipeline,
}

impl GpuNativeAttentionPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let read_only_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let read_write_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let rope_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_rope_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                ],
            });
        let kv_append_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_kv_append_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                ],
            });
        let causal_attention_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_causal_attention_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                ],
            });
        let rope_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_native_rope_pipeline_layout"),
            bind_group_layouts: &[&rope_bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..20,
            }],
        });
        let kv_append_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_native_kv_append_pipeline_layout"),
                bind_group_layouts: &[&kv_append_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..8,
                }],
            });
        let causal_attention_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_native_causal_attention_pipeline_layout"),
                bind_group_layouts: &[&causal_attention_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let rope_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_rope_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_ROPE_SHADER.into()),
        });
        let kv_append_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_kv_append_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_KV_APPEND_SHADER.into()),
        });
        let causal_attention_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_causal_attention_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_ATTENTION_SHADER.into()),
        });
        let rope = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_native_rope_pipeline"),
            layout: Some(&rope_pipeline_layout),
            module: &rope_module,
            entry_point: "rope_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let kv_append = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_native_kv_append_pipeline"),
            layout: Some(&kv_append_pipeline_layout),
            module: &kv_append_module,
            entry_point: "kv_append_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let causal_attention = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_native_causal_attention_pipeline"),
            layout: Some(&causal_attention_pipeline_layout),
            module: &causal_attention_module,
            entry_point: "causal_attention_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        Self {
            rope_bind_group_layout,
            kv_append_bind_group_layout,
            causal_attention_bind_group_layout,
            rope,
            kv_append,
            causal_attention,
        }
    }
}

/// Internal bootstrap for future GPU-owned token execution.
///
/// Retaining the exact `Arc<BackendBox>` keeps the authoritative non-cloneable
/// WGPU `Device` and `Queue` alive. It does not request or select hardware and
/// it is intentionally absent from all current execution-plan resolution.
pub(crate) struct GpuNativeExecutorContext {
    context_id: u64,
    authoritative_backend: Arc<BackendBox>,
    device_identity: GpuDeviceIdentity,
    layout: GpuNativeTokenStateLayout,
    dense_weights: ParkingMutex<GpuNativeDenseWeightRegistry>,
    dense_pipelines: GpuNativeDensePipelines,
    state_pipelines: GpuNativeStatePipelines,
    attention_pipelines: GpuNativeAttentionPipelines,
    counters: GpuNativeExecutionCounters,
}

impl GpuNativeExecutorContext {
    pub(super) fn try_new(
        authoritative_backend: Arc<BackendBox>,
        d_model: usize,
    ) -> Result<Self, GpuNativeBootstrapError> {
        let gpu = match authoritative_backend.as_ref() {
            BackendBox::Gpu(gpu) => gpu,
            BackendBox::Cpu(_) => return Err(GpuNativeBootstrapError::GpuBackendUnavailable),
            #[cfg(test)]
            BackendBox::TestGpu(_) => {
                return Err(GpuNativeBootstrapError::GpuBackendUnavailable);
            }
        };
        if let Some(detail) = gpu.device_loss.detail() {
            return Err(GpuNativeBootstrapError::DeviceLost { detail });
        }

        let layout = GpuNativeTokenStateLayout::try_new(d_model)?;
        layout.validate_for_limits(&gpu.device.limits())?;
        let device_identity = gpu.gpu_device_identity();
        let context_id = next_nonzero_id(&NEXT_GPU_NATIVE_CONTEXT_ID, "context");
        let dense_pipelines = GpuNativeDensePipelines::new(&gpu.device);
        let state_pipelines = GpuNativeStatePipelines::new(&gpu.device);
        let attention_pipelines = GpuNativeAttentionPipelines::new(&gpu.device);

        Ok(Self {
            context_id,
            authoritative_backend,
            device_identity,
            layout,
            dense_weights: ParkingMutex::new(GpuNativeDenseWeightRegistry::new(context_id)),
            dense_pipelines,
            state_pipelines,
            attention_pipelines,
            counters: GpuNativeExecutionCounters::default(),
        })
    }

    pub(crate) fn create_token_state(
        &self,
    ) -> Result<GpuNativeTokenState, GpuNativeBootstrapError> {
        let gpu = match self.authoritative_backend.as_ref() {
            BackendBox::Gpu(gpu) => gpu,
            BackendBox::Cpu(_) => return Err(GpuNativeBootstrapError::GpuBackendUnavailable),
            #[cfg(test)]
            BackendBox::TestGpu(_) => {
                return Err(GpuNativeBootstrapError::GpuBackendUnavailable);
            }
        };
        if let Some(detail) = gpu.device_loss.detail() {
            return Err(GpuNativeBootstrapError::DeviceLost { detail });
        }

        let state_id = next_gpu_native_token_state_id();
        let hidden = create_startup_buffer(
            &gpu.device,
            &format!("gpu_native_state_{state_id}_hidden"),
            self.layout.vector_bytes,
            GpuNativeTokenStateLayout::tensor_usage(),
        )?;
        let residual = create_startup_buffer(
            &gpu.device,
            &format!("gpu_native_state_{state_id}_residual"),
            self.layout.vector_bytes,
            GpuNativeTokenStateLayout::tensor_usage(),
        )?;
        let status = create_startup_buffer(
            &gpu.device,
            &format!("gpu_native_state_{state_id}_status"),
            self.layout.status_bytes,
            GpuNativeTokenStateLayout::status_usage(),
        )?;

        Ok(GpuNativeTokenState::from_buffers(
            self.context_id,
            state_id,
            self.layout,
            hidden,
            residual,
            status,
        ))
    }

    /// Register one immutable model-scoped dense tensor and upload its payload
    /// exactly once per physical chunk. This is the only dense-weight upload path in the
    /// GPU-native plane; encoded GEMV and embedding calls only bind this
    /// persistent model-scoped storage.
    pub(crate) fn register_dense_weight(
        &self,
        key: GpuNativeDenseWeightKey,
        weight: &DenseWeight,
    ) -> Result<GpuNativeDenseWeightHandle, GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        let layout = GpuNativeDenseWeightLayout::from_weight(weight)?;
        let plan = GpuNativeDenseWeightPlan::try_new(layout, &gpu.device.limits())?;

        // Serialize the duplicate check through insertion so two startup
        // registrars cannot both upload the same stable key.
        let mut registry = self.dense_weights.lock();
        if registry.weights.contains_key(&key) {
            return Err(GpuNativeBootstrapError::DuplicateDenseWeight {
                key: key.as_str().to_string(),
            });
        }
        for (index, chunk) in plan.chunks.iter().enumerate() {
            super::validate_startup_buffer(
                &format!("gpu_native_dense_weight_{}_chunk_{index}", key.as_str()),
                chunk.allocation_bytes,
                GpuNativeDenseWeightLayout::usage(),
                &gpu.device.limits(),
            )?;
        }

        let mut chunks = Vec::with_capacity(plan.chunks.len());
        for (index, chunk_plan) in plan.chunks.iter().copied().enumerate() {
            let label = format!("gpu_native_dense_weight_{}_chunk_{index}", key.as_str());
            let buffer = create_startup_buffer(
                &gpu.device,
                &label,
                chunk_plan.allocation_bytes,
                GpuNativeDenseWeightLayout::usage(),
            )?;
            match weight {
                DenseWeight::F32 { values, .. } => {
                    let value_start = chunk_plan.payload_offset_bytes / std::mem::size_of::<f32>();
                    let value_count =
                        chunk_plan.payload_bytes as usize / std::mem::size_of::<f32>();
                    gpu.queue.write_buffer(
                        &buffer,
                        0,
                        bytemuck::cast_slice(&values[value_start..value_start + value_count]),
                    );
                }
                DenseWeight::Q8_0 { bytes, .. }
                    if chunk_plan.payload_bytes == chunk_plan.allocation_bytes =>
                {
                    let start = chunk_plan.payload_offset_bytes;
                    let end = start + chunk_plan.payload_bytes as usize;
                    gpu.queue.write_buffer(&buffer, 0, &bytes[start..end]);
                }
                DenseWeight::Q8_0 { bytes, .. } => {
                    let start = chunk_plan.payload_offset_bytes;
                    let end = start + chunk_plan.payload_bytes as usize;
                    let mut upload = Vec::with_capacity(chunk_plan.allocation_bytes as usize);
                    upload.extend_from_slice(&bytes[start..end]);
                    upload.resize(chunk_plan.allocation_bytes as usize, 0);
                    gpu.queue.write_buffer(&buffer, 0, &upload);
                }
            }
            chunks.push(GpuNativeDenseWeightChunk {
                plan: chunk_plan,
                buffer,
            });
        }
        let registered = GpuNativeDenseWeight {
            weight_id: next_nonzero_id(&NEXT_GPU_NATIVE_WEIGHT_ID, "dense weight"),
            key,
            layout,
            chunks,
        };
        let handle = registry.insert(registered)?;
        self.counters.record_dense_weight_registration(
            plan.chunks.len() as u64,
            plan.physical_allocation_bytes,
        );
        Ok(handle)
    }

    /// Register one immutable RMSNorm gain vector in the existing persistent
    /// F32 dense-weight registry. The typed handle deliberately exposes no
    /// matrix interface.
    pub(crate) fn register_rms_norm(
        &self,
        key: GpuNativeDenseWeightKey,
        weight: &[f32],
    ) -> Result<GpuNativeRmsNormHandle, GpuNativeBootstrapError> {
        validate_rms_norm_weight_width(weight.len())?;
        let dense = DenseWeight::from_f32(weight.to_vec(), 1, weight.len());
        self.register_dense_weight(key, &dense)
            .map(GpuNativeRmsNormHandle::from_dense)
    }

    /// Register already-derived model-scoped RoPE inverse frequencies. This
    /// reuses the persistent dense F32 registry and performs no per-token
    /// parameter upload.
    pub(crate) fn register_rope_parameters(
        &self,
        key: GpuNativeDenseWeightKey,
        rope_dim: usize,
        inverse_frequencies: &[f32],
        attention_factor: f32,
    ) -> Result<GpuNativeRopeHandle, GpuNativeBootstrapError> {
        let layout = GpuNativeRopeLayout::try_new(rope_dim, rope_dim)?;
        validate_rope_parameters(layout, inverse_frequencies, attention_factor)?;
        let dense =
            DenseWeight::from_f32(inverse_frequencies.to_vec(), 1, inverse_frequencies.len());
        let dense = self.register_dense_weight(key, &dense)?;
        self.counters
            .record_rope_registration(dense.layout.allocation_bytes);
        Ok(GpuNativeRopeHandle {
            dense,
            rope_dim,
            attention_factor_bits: attention_factor.to_bits(),
        })
    }

    /// Register the standard Qwen/Llama inverse-frequency schedule.
    pub(crate) fn register_standard_rope(
        &self,
        key: GpuNativeDenseWeightKey,
        rope_dim: usize,
        base: f32,
    ) -> Result<GpuNativeRopeHandle, GpuNativeBootstrapError> {
        let inverse_frequencies = standard_rope_inverse_frequencies(rope_dim, base)?;
        self.register_rope_parameters(key, rope_dim, &inverse_frequencies, 1.0)
    }

    pub(crate) fn dense_weight_handle(
        &self,
        key: &GpuNativeDenseWeightKey,
        expected_kind: GpuNativeDenseWeightKind,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<GpuNativeDenseWeightHandle, GpuNativeBootstrapError> {
        self.dense_weights
            .lock()
            .handle_for(key, expected_kind, expected_rows, expected_cols)
    }

    pub(crate) fn rms_norm_handle(
        &self,
        key: &GpuNativeDenseWeightKey,
        expected_width: usize,
    ) -> Result<GpuNativeRmsNormHandle, GpuNativeBootstrapError> {
        validate_rms_norm_weight_width(expected_width)?;
        self.dense_weights
            .lock()
            .handle_for(key, GpuNativeDenseWeightKind::F32, 1, expected_width)
            .map(GpuNativeRmsNormHandle::from_dense)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_attention_plan(
        &self,
        layer_index: usize,
        geometry: GpuNativeAttentionGeometry,
        q_projection: GpuNativeDenseWeightHandle,
        k_projection: GpuNativeDenseWeightHandle,
        v_projection: GpuNativeDenseWeightHandle,
        o_projection: GpuNativeDenseWeightHandle,
        q_norm: Option<GpuNativeAttentionNorm>,
        k_norm: Option<GpuNativeAttentionNorm>,
        rope: GpuNativeRopeHandle,
    ) -> Result<GpuNativeAttentionPlan, GpuNativeBootstrapError> {
        let plan = GpuNativeAttentionPlan {
            context_id: self.context_id,
            layer_index,
            geometry,
            q_projection,
            k_projection,
            v_projection,
            o_projection,
            q_norm,
            k_norm,
            rope,
        };
        self.validate_attention_plan(&plan)?;
        Ok(plan)
    }

    pub(crate) fn create_scratch(
        &self,
        elements: usize,
    ) -> Result<GpuNativeScratch, GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        let layout = GpuNativeScratchLayout::try_new(elements)?;
        layout.validate_for_limits(&gpu.device.limits())?;
        let scratch_id = next_nonzero_id(&NEXT_GPU_NATIVE_SCRATCH_ID, "scratch");
        let buffer = create_startup_buffer(
            &gpu.device,
            &format!("gpu_native_scratch_{scratch_id}"),
            layout.bytes,
            GpuNativeScratchLayout::usage(),
        )?;
        Ok(GpuNativeScratch::from_buffer(
            self.context_id,
            scratch_id,
            layout,
            buffer,
        ))
    }

    pub(crate) fn create_attention_scratch(
        &self,
        geometry: GpuNativeAttentionGeometry,
    ) -> Result<GpuNativeAttentionScratch, GpuNativeBootstrapError> {
        if geometry.d_model != self.layout.d_model {
            return Err(GpuNativeBootstrapError::AttentionDModelMismatch {
                expected: self.layout.d_model,
                actual: geometry.d_model,
            });
        }
        let gpu = self.authoritative_gpu()?;
        for elements in [
            geometry.q_width,
            geometry.kv_width,
            geometry.kv_width,
            geometry.q_width,
            geometry.d_model,
        ] {
            GpuNativeScratchLayout::try_new(elements)?.validate_for_limits(&gpu.device.limits())?;
        }
        let q = self.create_scratch(geometry.q_width)?;
        let k = self.create_scratch(geometry.kv_width)?;
        let v = self.create_scratch(geometry.kv_width)?;
        let context = self.create_scratch(geometry.q_width)?;
        let projected = self.create_scratch(geometry.d_model)?;
        Ok(GpuNativeAttentionScratch::from_scratch(
            self.context_id,
            geometry,
            q,
            k,
            v,
            context,
            projected,
        ))
    }

    pub(crate) fn create_kv_state(
        &self,
        num_layers: usize,
        max_seq_len: usize,
        kv_width: usize,
    ) -> Result<GpuNativeKvState, GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        let layout =
            GpuNativeKvLayout::try_new(num_layers, max_seq_len, kv_width, &gpu.device.limits())?;
        let kv_id = next_nonzero_id(&NEXT_GPU_NATIVE_KV_ID, "KV state");
        let mut layers = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let key = create_startup_buffer(
                &gpu.device,
                &format!("gpu_native_kv_{kv_id}_layer_{layer}_key"),
                layout.layer_bytes,
                GpuNativeKvLayout::usage(),
            )?;
            let value = create_startup_buffer(
                &gpu.device,
                &format!("gpu_native_kv_{kv_id}_layer_{layer}_value"),
                layout.layer_bytes,
                GpuNativeKvLayout::usage(),
            )?;
            layers.push(GpuNativeKvLayer { key, value });
        }
        Ok(GpuNativeKvState::from_layers(
            self.context_id,
            kv_id,
            layout,
            layers,
        ))
    }

    fn validate_attention_plan(
        &self,
        plan: &GpuNativeAttentionPlan,
    ) -> Result<(), GpuNativeBootstrapError> {
        let registry = self.dense_weights.lock();
        validate_attention_plan_with_registry(self.context_id, self.layout.d_model, &registry, plan)
    }

    /// Encode `weight[rows, cols] * state.hidden[cols] -> output[rows]`.
    pub(crate) fn encode_dense_gemv_hidden_to_scratch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeDenseWeightHandle,
        state: &GpuNativeTokenState,
        output: &GpuNativeScratch,
    ) -> Result<(), GpuNativeBootstrapError> {
        if state.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignTokenState);
        }
        if output.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignScratch);
        }
        self.encode_dense_gemv_buffers(
            encoder,
            handle,
            &state.hidden,
            state.layout.d_model,
            &output.buffer,
            output.layout.elements,
        )
    }

    /// Encode `weight * input -> output` between distinct request scratch
    /// buffers without exposing either raw WGPU buffer outside this module.
    pub(crate) fn encode_dense_gemv_scratch_to_scratch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeDenseWeightHandle,
        input: &GpuNativeScratch,
        output: &GpuNativeScratch,
    ) -> Result<(), GpuNativeBootstrapError> {
        if input.context_id != self.context_id || output.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignScratch);
        }
        if input.scratch_id == output.scratch_id {
            return Err(GpuNativeBootstrapError::AliasedInputOutput);
        }
        self.encode_dense_gemv_buffers(
            encoder,
            handle,
            &input.buffer,
            input.layout.elements,
            &output.buffer,
            output.layout.elements,
        )
    }

    /// Encode `weight * input -> state.hidden` for a matrix whose row count is
    /// exactly d_model.
    pub(crate) fn encode_dense_gemv_scratch_to_hidden(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeDenseWeightHandle,
        input: &GpuNativeScratch,
        state: &GpuNativeTokenState,
    ) -> Result<(), GpuNativeBootstrapError> {
        if input.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignScratch);
        }
        if state.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignTokenState);
        }
        self.encode_dense_gemv_buffers(
            encoder,
            handle,
            &input.buffer,
            input.layout.elements,
            &state.hidden,
            state.layout.d_model,
        )
    }

    fn encode_dense_gemv_buffers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeDenseWeightHandle,
        input: &wgpu::Buffer,
        input_elements: usize,
        output: &wgpu::Buffer,
        output_elements: usize,
    ) -> Result<(), GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        let weight = self.dense_weights.lock().resolve(handle)?;
        if input_elements != weight.layout.cols {
            return Err(GpuNativeBootstrapError::GemvInputLength {
                expected: weight.layout.cols,
                actual: input_elements,
            });
        }
        if output_elements != weight.layout.rows {
            return Err(GpuNativeBootstrapError::GemvOutputLength {
                expected: weight.layout.rows,
                actual: output_elements,
            });
        }
        let workgroups = weight
            .chunks
            .iter()
            .map(|chunk| self.checked_workgroups(chunk.plan.row_count, &gpu.device.limits()))
            .collect::<Result<Vec<_>, _>>()?;
        self.encode_dense_gemv_resolved(gpu, encoder, &weight, input, output, &workgroups);
        Ok(())
    }

    fn encode_dense_gemv_resolved(
        &self,
        gpu: &super::GpuBackend,
        encoder: &mut wgpu::CommandEncoder,
        weight: &GpuNativeDenseWeight,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        workgroups: &[u32],
    ) {
        let pipeline = match weight.layout.kind {
            GpuNativeDenseWeightKind::F32 => &self.dense_pipelines.f32_gemv,
            GpuNativeDenseWeightKind::Q8_0 => &self.dense_pipelines.q8_0_gemv,
        };
        for (chunk, workgroups) in weight.chunks.iter().zip(workgroups.iter().copied()) {
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu_native_dense_gemv_chunk_bind_group"),
                layout: &self.dense_pipelines.gemv_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: chunk.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_native_dense_gemv_chunk_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_push_constants(
                0,
                bytemuck::bytes_of(&GpuNativeGemvPushConstants {
                    rows: chunk.plan.row_count as u32,
                    cols: weight.layout.cols as u32,
                    global_row_base: chunk.plan.row_start as u32,
                    q8_first_block: chunk.plan.first_block as u32,
                }),
            );
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.counters
            .record_dense_gemv_dispatch(weight.chunks.len() as u64);
    }

    /// Encode one embedding row directly into request-local hidden state.
    pub(crate) fn encode_embedding_lookup(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeDenseWeightHandle,
        token_id: u32,
        state: &GpuNativeTokenState,
    ) -> Result<(), GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        if state.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignTokenState);
        }
        let weight = self.dense_weights.lock().resolve(handle)?;
        weight.layout.validate_embedding_token(token_id)?;
        if weight.layout.cols != state.layout.d_model {
            return Err(GpuNativeBootstrapError::EmbeddingWidth {
                expected: state.layout.d_model,
                actual: weight.layout.cols,
            });
        }
        let workgroups = self.checked_workgroups(weight.layout.cols, &gpu.device.limits())?;
        let chunk = weight
            .chunks
            .iter()
            .find(|chunk| chunk.plan.contains_row(token_id as usize))
            .expect("validated embedding row must belong to exactly one chunk");
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_embedding_bind_group"),
            layout: &self.dense_pipelines.embedding_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: chunk.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state.hidden.as_entire_binding(),
                },
            ],
        });
        let pipeline = match weight.layout.kind {
            GpuNativeDenseWeightKind::F32 => &self.dense_pipelines.f32_embedding,
            GpuNativeDenseWeightKind::Q8_0 => &self.dense_pipelines.q8_0_embedding,
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_embedding_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(
            0,
            bytemuck::bytes_of(&GpuNativeEmbeddingPushConstants {
                local_row: token_id - chunk.plan.row_start as u32,
                global_row: token_id,
                cols: weight.layout.cols as u32,
                q8_first_block: chunk.plan.first_block as u32,
            }),
        );
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
        self.counters.record_embedding_dispatch();
        Ok(())
    }

    /// Preserve the current residual stream and RMS-normalise hidden in one
    /// dispatch: `residual = old hidden`, `hidden = rms_norm(old hidden)`.
    pub(crate) fn encode_rms_norm_state_in_place(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeRmsNormHandle,
        epsilon: f32,
        state: &GpuNativeTokenState,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_token_state_owner(self.context_id, state.context_id)?;
        self.encode_rms_norm_buffers(
            encoder,
            handle,
            epsilon,
            &state.hidden,
            state.layout.d_model,
            Some(&state.residual),
            1,
            state.layout.d_model,
            false,
        )
    }

    /// Apply final-model RMSNorm to hidden without changing the saved residual.
    pub(crate) fn encode_rms_norm_hidden_in_place(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeRmsNormHandle,
        epsilon: f32,
        state: &GpuNativeTokenState,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_token_state_owner(self.context_id, state.context_id)?;
        self.encode_rms_norm_buffers(
            encoder,
            handle,
            epsilon,
            &state.hidden,
            state.layout.d_model,
            None,
            1,
            state.layout.d_model,
            false,
        )
    }

    /// RMS-normalise each logical scratch group independently using a shared
    /// `group_width`-element gain vector.
    pub(crate) fn encode_rms_norm_scratch_in_place(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeRmsNormHandle,
        epsilon: f32,
        scratch: &GpuNativeScratch,
        groups: usize,
        group_width: usize,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_scratch_owner(self.context_id, scratch.context_id)?;
        self.encode_rms_norm_buffers(
            encoder,
            handle,
            epsilon,
            &scratch.buffer,
            scratch.layout.elements,
            None,
            groups,
            group_width,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_rms_norm_buffers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeRmsNormHandle,
        epsilon: f32,
        target: &wgpu::Buffer,
        target_elements: usize,
        residual: Option<&wgpu::Buffer>,
        groups: usize,
        group_width: usize,
        scratch_dispatch: bool,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_rms_norm_epsilon(epsilon)?;
        let gpu = self.authoritative_gpu()?;
        let weight = self.dense_weights.lock().resolve_rms_norm(handle)?;
        let geometry = GpuNativeRmsNormGeometry::try_new(
            groups,
            group_width,
            target_elements,
            weight.layout.cols,
        )?;
        let workgroups = geometry.checked_workgroups(&gpu.device.limits())?;
        let chunk = match weight.chunks.as_slice() {
            [chunk] => chunk,
            _ => {
                return Err(GpuNativeBootstrapError::StaleRmsNormHandle {
                    key: handle.dense.key.as_str().to_string(),
                });
            }
        };
        let push_constants = GpuNativeRmsNormPushConstants {
            groups: geometry.groups as u32,
            group_width: geometry.group_width as u32,
            epsilon_bits: epsilon.to_bits(),
            _reserved: 0,
        };

        match residual {
            Some(residual) => {
                let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gpu_native_rms_capture_bind_group"),
                    layout: &self.state_pipelines.rms_capture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: chunk.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: target.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: residual.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_native_rms_capture_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.state_pipelines.rms_capture);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_push_constants(0, bytemuck::bytes_of(&push_constants));
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            None => {
                let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gpu_native_rms_in_place_bind_group"),
                    layout: &self.state_pipelines.rms_in_place_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: chunk.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: target.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_native_rms_in_place_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.state_pipelines.rms_in_place);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_push_constants(0, bytemuck::bytes_of(&push_constants));
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
        if scratch_dispatch {
            self.counters
                .record_rms_norm_scratch_dispatch(geometry.groups as u64);
        } else {
            self.counters
                .record_rms_norm_state_dispatch(geometry.groups as u64);
        }
        Ok(())
    }

    fn validate_attention_dispatch_limits(
        &self,
        plan: &GpuNativeAttentionPlan,
        limits: &wgpu::Limits,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_causal_attention_dispatch(plan.geometry, limits)?;
        let registry = self.dense_weights.lock();
        for handle in [
            &plan.q_projection,
            &plan.k_projection,
            &plan.v_projection,
            &plan.o_projection,
        ] {
            let weight = registry.resolve(handle)?;
            for chunk in &weight.chunks {
                self.checked_workgroups(chunk.plan.row_count, limits)?;
            }
        }
        for groups in [plan.geometry.num_heads, plan.geometry.num_kv_heads] {
            if groups as u64 > limits.max_compute_workgroups_per_dimension as u64 {
                return Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
                    workgroups: groups as u64,
                    maximum: limits.max_compute_workgroups_per_dimension,
                });
            }
            let pairs = groups.checked_mul(plan.geometry.rope_dim / 2).ok_or(
                GpuNativeBootstrapError::AttentionGeometryOverflow {
                    num_heads: plan.geometry.num_heads,
                    num_kv_heads: plan.geometry.num_kv_heads,
                    head_dim: plan.geometry.head_dim,
                },
            )?;
            self.checked_workgroups(pairs, limits)?;
        }
        self.checked_workgroups(plan.geometry.kv_width, limits)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_rope_scratch_in_place(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeRopeHandle,
        scratch: &GpuNativeScratch,
        tensor: GpuNativeAttentionTensor,
        groups: usize,
        head_dim: usize,
        position: usize,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_scratch_owner(self.context_id, scratch.context_id)?;
        validate_rope_dimension(handle.rope_dim, head_dim)?;
        let expected = groups.checked_mul(head_dim).ok_or(
            GpuNativeBootstrapError::AttentionGeometryOverflow {
                num_heads: groups,
                num_kv_heads: groups,
                head_dim,
            },
        )?;
        if scratch.layout.elements != expected {
            return Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor,
                expected,
                actual: scratch.layout.elements,
            });
        }
        let position =
            u32::try_from(position).map_err(|_| GpuNativeBootstrapError::InvalidKvPosition {
                position,
                max_seq_len: u32::MAX as usize,
            })?;
        let gpu = self.authoritative_gpu()?;
        let registry = self.dense_weights.lock();
        validate_rope_handle_with_registry(self.context_id, &registry, handle, handle.rope_dim)?;
        let weight = registry.resolve(&handle.dense)?;
        let chunk = match weight.chunks.as_slice() {
            [chunk] => chunk,
            _ => {
                return Err(GpuNativeBootstrapError::StaleRopeHandle {
                    key: handle.dense.key.as_str().to_string(),
                });
            }
        };
        let pairs = groups.checked_mul(handle.rope_dim / 2).ok_or(
            GpuNativeBootstrapError::AttentionGeometryOverflow {
                num_heads: groups,
                num_kv_heads: groups,
                head_dim,
            },
        )?;
        let workgroups = self.checked_workgroups(pairs, &gpu.device.limits())?;
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_rope_bind_group"),
            layout: &self.attention_pipelines.rope_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: chunk.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scratch.buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_rope_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.attention_pipelines.rope);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(
            0,
            bytemuck::bytes_of(&GpuNativeRopePushConstants {
                groups: groups as u32,
                head_dim: head_dim as u32,
                rope_dim: handle.rope_dim as u32,
                position,
                attention_factor_bits: handle.attention_factor_bits,
            }),
        );
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
        self.counters.record_rope_dispatch(groups as u64);
        Ok(())
    }

    fn encode_kv_append(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        k: &GpuNativeScratch,
        v: &GpuNativeScratch,
        kv: &GpuNativeKvState,
        layer: usize,
        position: usize,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_scratch_owner(self.context_id, k.context_id)?;
        validate_scratch_owner(self.context_id, v.context_id)?;
        validate_kv_state(self.context_id, k.layout.elements, kv, layer, position)?;
        if v.layout.elements != kv.layout.kv_width {
            return Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor: GpuNativeAttentionTensor::Value,
                expected: kv.layout.kv_width,
                actual: v.layout.elements,
            });
        }
        let gpu = self.authoritative_gpu()?;
        let workgroups = self.checked_workgroups(kv.layout.kv_width, &gpu.device.limits())?;
        let layer_buffers = &kv.layers[layer];
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_kv_append_bind_group"),
            layout: &self.attention_pipelines.kv_append_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: k.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: v.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: layer_buffers.key.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: layer_buffers.value.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_kv_append_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.attention_pipelines.kv_append);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(
            0,
            bytemuck::bytes_of(&GpuNativeKvAppendPushConstants {
                width: kv.layout.kv_width as u32,
                position: position as u32,
            }),
        );
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
        self.counters.record_kv_append();
        Ok(())
    }

    /// Compose Q/K/V projection, optional per-head QK-Norm, per-head RoPE,
    /// and absolute-position request-local KV append into the caller's encoder.
    pub(crate) fn encode_attention_prepare(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &GpuNativeAttentionPlan,
        state: &GpuNativeTokenState,
        scratch: &GpuNativeAttentionScratch,
        kv: &GpuNativeKvState,
        position: usize,
    ) -> Result<(), GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        self.validate_attention_plan(plan)?;
        validate_token_state_owner(self.context_id, state.context_id)?;
        if state.layout.d_model != plan.geometry.d_model {
            return Err(GpuNativeBootstrapError::AttentionDModelMismatch {
                expected: plan.geometry.d_model,
                actual: state.layout.d_model,
            });
        }
        validate_attention_scratch(self.context_id, plan.geometry, scratch)?;
        validate_attention_kv_state(
            self.context_id,
            plan.geometry,
            kv,
            plan.layer_index,
            position,
        )?;
        self.validate_attention_dispatch_limits(plan, &gpu.device.limits())?;

        self.encode_dense_gemv_hidden_to_scratch(encoder, &plan.q_projection, state, &scratch.q)?;
        self.encode_dense_gemv_hidden_to_scratch(encoder, &plan.k_projection, state, &scratch.k)?;
        self.encode_dense_gemv_hidden_to_scratch(encoder, &plan.v_projection, state, &scratch.v)?;
        if let Some(norm) = &plan.q_norm {
            self.encode_rms_norm_scratch_in_place(
                encoder,
                &norm.handle,
                norm.epsilon(),
                &scratch.q,
                plan.geometry.num_heads,
                plan.geometry.head_dim,
            )?;
        }
        if let Some(norm) = &plan.k_norm {
            self.encode_rms_norm_scratch_in_place(
                encoder,
                &norm.handle,
                norm.epsilon(),
                &scratch.k,
                plan.geometry.num_kv_heads,
                plan.geometry.head_dim,
            )?;
        }
        self.encode_rope_scratch_in_place(
            encoder,
            &plan.rope,
            &scratch.q,
            GpuNativeAttentionTensor::Query,
            plan.geometry.num_heads,
            plan.geometry.head_dim,
            position,
        )?;
        self.encode_rope_scratch_in_place(
            encoder,
            &plan.rope,
            &scratch.k,
            GpuNativeAttentionTensor::Key,
            plan.geometry.num_kv_heads,
            plan.geometry.head_dim,
            position,
        )?;
        self.encode_kv_append(
            encoder,
            &scratch.k,
            &scratch.v,
            kv,
            plan.layer_index,
            position,
        )?;
        self.counters.record_attention_prepare_dispatch();
        Ok(())
    }

    fn encode_causal_attention_pass(
        &self,
        gpu: &super::GpuBackend,
        encoder: &mut wgpu::CommandEncoder,
        plan: &GpuNativeAttentionPlan,
        scratch: &GpuNativeAttentionScratch,
        kv: &GpuNativeKvState,
        seq_len: u32,
    ) {
        let layer_buffers = &kv.layers[plan.layer_index];
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_causal_attention_bind_group"),
            layout: &self.attention_pipelines.causal_attention_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scratch.q.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: layer_buffers.key.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: layer_buffers.value.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch.context.buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_causal_attention_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.attention_pipelines.causal_attention);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(
            0,
            bytemuck::bytes_of(&GpuNativeAttentionPushConstants {
                num_heads: plan.geometry.num_heads as u32,
                num_kv_heads: plan.geometry.num_kv_heads as u32,
                head_dim: plan.geometry.head_dim as u32,
                seq_len,
            }),
        );
        pass.dispatch_workgroups(plan.geometry.num_heads as u32, 1, 1);
        drop(pass);
        self.counters.record_causal_attention_dispatch();
    }

    /// Complete one prepared incremental attention operation entirely in the
    /// caller's encoder: causal attention, persistent O projection, then the
    /// saved pre-attention residual add. `state.residual` is never a target.
    pub(crate) fn encode_attention_complete(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &GpuNativeAttentionPlan,
        state: &GpuNativeTokenState,
        scratch: &GpuNativeAttentionScratch,
        kv: &GpuNativeKvState,
        position: usize,
    ) -> Result<(), GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        self.validate_attention_plan(plan)?;
        validate_token_state_owner(self.context_id, state.context_id)?;
        if state.layout.d_model != plan.geometry.d_model {
            return Err(GpuNativeBootstrapError::AttentionDModelMismatch {
                expected: plan.geometry.d_model,
                actual: state.layout.d_model,
            });
        }
        validate_attention_scratch(self.context_id, plan.geometry, scratch)?;
        let seq_len = validate_attention_kv_state(
            self.context_id,
            plan.geometry,
            kv,
            plan.layer_index,
            position,
        )?;
        let seq_len = u32::try_from(seq_len).map_err(|_| {
            GpuNativeBootstrapError::InvalidAttentionSequenceLength {
                seq_len,
                max_seq_len: kv.layout.max_seq_len,
            }
        })?;
        self.validate_attention_dispatch_limits(plan, &gpu.device.limits())?;
        validate_residual_contribution_width(
            state.layout.d_model,
            scratch.projected.layout.elements,
        )?;

        // Resolve and validate every fallible O-projection property before the
        // first completion command is recorded. The registry never removes a
        // weight, so the Arc remains authoritative for the encoded passes.
        let o_projection = self.dense_weights.lock().resolve(&plan.o_projection)?;
        if scratch.context.layout.elements != o_projection.layout.cols {
            return Err(GpuNativeBootstrapError::GemvInputLength {
                expected: o_projection.layout.cols,
                actual: scratch.context.layout.elements,
            });
        }
        if scratch.projected.layout.elements != o_projection.layout.rows {
            return Err(GpuNativeBootstrapError::GemvOutputLength {
                expected: o_projection.layout.rows,
                actual: scratch.projected.layout.elements,
            });
        }
        let o_workgroups = o_projection
            .chunks
            .iter()
            .map(|chunk| self.checked_workgroups(chunk.plan.row_count, &gpu.device.limits()))
            .collect::<Result<Vec<_>, _>>()?;
        let residual_workgroups =
            self.checked_workgroups(state.layout.d_model, &gpu.device.limits())?;

        self.encode_causal_attention_pass(gpu, encoder, plan, scratch, kv, seq_len);
        self.encode_dense_gemv_resolved(
            gpu,
            encoder,
            &o_projection,
            &scratch.context.buffer,
            &scratch.projected.buffer,
            &o_workgroups,
        );
        self.encode_residual_add_pass(gpu, encoder, state, &scratch.projected, residual_workgroups);
        self.counters.record_attention_complete_dispatch();
        Ok(())
    }

    /// Complete a prepared sub-block entirely on device:
    /// `hidden = residual + contribution`.
    pub(crate) fn encode_residual_add_scratch_to_hidden(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        state: &GpuNativeTokenState,
        contribution: &GpuNativeScratch,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_token_state_owner(self.context_id, state.context_id)?;
        validate_scratch_owner(self.context_id, contribution.context_id)?;
        validate_residual_contribution_width(state.layout.d_model, contribution.layout.elements)?;
        let gpu = self.authoritative_gpu()?;
        let workgroups = self.checked_workgroups(state.layout.d_model, &gpu.device.limits())?;
        self.encode_residual_add_pass(gpu, encoder, state, contribution, workgroups);
        Ok(())
    }

    fn encode_residual_add_pass(
        &self,
        gpu: &super::GpuBackend,
        encoder: &mut wgpu::CommandEncoder,
        state: &GpuNativeTokenState,
        contribution: &GpuNativeScratch,
        workgroups: u32,
    ) {
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_residual_add_bind_group"),
            layout: &self.state_pipelines.rms_capture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: contribution.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state.hidden.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.residual.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_residual_add_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.state_pipelines.residual_add);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(
            0,
            bytemuck::bytes_of(&GpuNativeRmsNormPushConstants {
                groups: 1,
                group_width: state.layout.d_model as u32,
                epsilon_bits: 0,
                _reserved: 0,
            }),
        );
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
        self.counters.record_residual_add_dispatch();
    }

    fn authoritative_gpu(&self) -> Result<&super::GpuBackend, GpuNativeBootstrapError> {
        let gpu = match self.authoritative_backend.as_ref() {
            BackendBox::Gpu(gpu) => gpu,
            BackendBox::Cpu(_) => return Err(GpuNativeBootstrapError::GpuBackendUnavailable),
            #[cfg(test)]
            BackendBox::TestGpu(_) => {
                return Err(GpuNativeBootstrapError::GpuBackendUnavailable);
            }
        };
        if let Some(detail) = gpu.device_loss.detail() {
            return Err(GpuNativeBootstrapError::DeviceLost { detail });
        }
        Ok(gpu)
    }

    fn checked_workgroups(
        &self,
        elements: usize,
        limits: &wgpu::Limits,
    ) -> Result<u32, GpuNativeBootstrapError> {
        let elements = u64::try_from(elements).map_err(|_| {
            GpuNativeBootstrapError::DispatchGeometryUnsupported {
                workgroups: u64::MAX,
                maximum: limits.max_compute_workgroups_per_dimension,
            }
        })?;
        let workgroups = elements.div_ceil(GPU_NATIVE_WORKGROUP_SIZE as u64);
        if workgroups > limits.max_compute_workgroups_per_dimension as u64 {
            return Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
                workgroups,
                maximum: limits.max_compute_workgroups_per_dimension,
            });
        }
        Ok(workgroups as u32)
    }

    pub(crate) fn device_identity(&self) -> &GpuDeviceIdentity {
        &self.device_identity
    }

    pub(crate) const fn token_state_layout(&self) -> GpuNativeTokenStateLayout {
        self.layout
    }

    pub(crate) fn execution_snapshot(&self) -> GpuNativeExecutionSnapshot {
        self.counters.snapshot()
    }
}

impl fmt::Debug for GpuNativeExecutorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuNativeExecutorContext")
            .field("device_identity", &self.device_identity)
            .field("layout", &self.layout)
            .field("snapshot", &self.execution_snapshot())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{dequantize_q8_0_block, quantize_q8_0_block};
    use std::sync::atomic::AtomicUsize;

    fn q8_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = vec![0; values.len().div_ceil(Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES];
        for (block, chunk) in values.chunks(Q8_0_BLOCK_ELEMS).enumerate() {
            quantize_q8_0_block(
                chunk,
                &mut bytes[block * Q8_0_BLOCK_BYTES..(block + 1) * Q8_0_BLOCK_BYTES],
            );
        }
        bytes
    }

    fn read_q8_mirror(bytes: &[u8], flat_index: usize) -> f32 {
        let block = flat_index / Q8_0_BLOCK_ELEMS;
        let in_block = flat_index % Q8_0_BLOCK_ELEMS;
        let offset = block * Q8_0_BLOCK_BYTES;
        let scale = half::f16::from_le_bytes([bytes[offset], bytes[offset + 1]]).to_f32();
        scale * (bytes[offset + 2 + in_block] as i8 as f32)
    }

    fn read_q8_chunk_mirror(
        bytes: &[u8],
        plan: GpuNativeDenseWeightChunkPlan,
        global_flat_index: usize,
    ) -> f32 {
        let global_block = global_flat_index / Q8_0_BLOCK_ELEMS;
        let local_block = global_block - plan.first_block;
        let in_block = global_flat_index % Q8_0_BLOCK_ELEMS;
        let offset = local_block * Q8_0_BLOCK_BYTES;
        let scale = half::f16::from_le_bytes([bytes[offset], bytes[offset + 1]]).to_f32();
        scale * (bytes[offset + 2 + in_block] as i8 as f32)
    }

    fn q8_gemv_mirror(bytes: &[u8], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
        (0..rows)
            .map(|row| {
                let mut sum = 0.0;
                for col in 0..cols {
                    sum += read_q8_mirror(bytes, row * cols + col) * input[col];
                }
                sum
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    fn rms_norm_mirror(
        values: &[f32],
        weight: &[f32],
        epsilon: f32,
        groups: usize,
        group_width: usize,
    ) -> Vec<f32> {
        assert_eq!(values.len(), groups * group_width);
        assert_eq!(weight.len(), group_width);
        let mut result = values.to_vec();
        for group in result.chunks_exact_mut(group_width) {
            let mut squared_sum = 0.0f32;
            for &value in group.iter() {
                squared_sum += value * value;
            }
            let mean_square = squared_sum / group_width as f32;
            let inverse_rms = 1.0 / (mean_square + epsilon).sqrt();
            for (value, gain) in group.iter_mut().zip(weight) {
                *value = *value * inverse_rms * *gain;
            }
        }
        result
    }

    fn residual_add_mirror(residual: &[f32], contribution: &[f32]) -> Vec<f32> {
        residual
            .iter()
            .zip(contribution)
            .map(|(&residual, &contribution)| residual + contribution)
            .collect()
    }

    fn rms_norm_capture_mirror(
        hidden: &mut Vec<f32>,
        residual: &mut Vec<f32>,
        weight: &[f32],
        epsilon: f32,
    ) {
        residual.clone_from(hidden);
        let width = hidden.len();
        *hidden = rms_norm_mirror(hidden, weight, epsilon, 1, width);
    }

    fn residual_complete_mirror(hidden: &mut Vec<f32>, residual: &[f32], contribution: &[f32]) {
        *hidden = residual_add_mirror(residual, contribution);
    }

    fn rope_mirror(
        values: &[f32],
        groups: usize,
        head_dim: usize,
        rope_dim: usize,
        position: usize,
        inverse_frequencies: &[f32],
        attention_factor: f32,
    ) -> Vec<f32> {
        assert_eq!(values.len(), groups * head_dim);
        assert_eq!(inverse_frequencies.len(), rope_dim / 2);
        let mut result = values.to_vec();
        let pairs = rope_dim / 2;
        for group in 0..groups {
            let head_start = group * head_dim;
            for pair in 0..pairs {
                let theta = position as f32 * inverse_frequencies[pair];
                let (sin_theta, cos_theta) = theta.sin_cos();
                let sin_theta = sin_theta * attention_factor;
                let cos_theta = cos_theta * attention_factor;
                let first = head_start + pair;
                let second = first + pairs;
                let a = result[first];
                let b = result[second];
                result[first] = a * cos_theta - b * sin_theta;
                result[second] = a * sin_theta + b * cos_theta;
            }
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_prepare_mirror(
        hidden: &[f32],
        geometry: GpuNativeAttentionGeometry,
        q_projection: &DenseWeight,
        k_projection: &DenseWeight,
        v_projection: &DenseWeight,
        q_norm: Option<(&[f32], f32)>,
        k_norm: Option<(&[f32], f32)>,
        inverse_frequencies: &[f32],
        attention_factor: f32,
        position: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut q = q_projection.matvec(hidden);
        let mut k = k_projection.matvec(hidden);
        let v = v_projection.matvec(hidden);
        if let Some((gain, epsilon)) = q_norm {
            q = rms_norm_mirror(&q, gain, epsilon, geometry.num_heads, geometry.head_dim);
        }
        if let Some((gain, epsilon)) = k_norm {
            k = rms_norm_mirror(&k, gain, epsilon, geometry.num_kv_heads, geometry.head_dim);
        }
        q = rope_mirror(
            &q,
            geometry.num_heads,
            geometry.head_dim,
            geometry.rope_dim,
            position,
            inverse_frequencies,
            attention_factor,
        );
        k = rope_mirror(
            &k,
            geometry.num_kv_heads,
            geometry.head_dim,
            geometry.rope_dim,
            position,
            inverse_frequencies,
            attention_factor,
        );
        (q, k, v)
    }

    fn causal_attention_mirror(
        q: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        geometry: GpuNativeAttentionGeometry,
        seq_len: usize,
    ) -> Vec<f32> {
        assert_eq!(q.len(), geometry.q_width);
        assert!(seq_len > 0);
        assert!(key_cache.len() >= seq_len * geometry.kv_width);
        assert!(value_cache.len() >= seq_len * geometry.kv_width);
        let mut context = vec![0.0; geometry.q_width];
        let scale = 1.0 / (geometry.head_dim as f32).sqrt();
        for query_head in 0..geometry.num_heads {
            let kv_head = query_head * geometry.num_kv_heads / geometry.num_heads;
            let q_start = query_head * geometry.head_dim;
            let q_head = &q[q_start..q_start + geometry.head_dim];
            let mut scores = Vec::with_capacity(seq_len);
            for position in 0..seq_len {
                let k_start = position * geometry.kv_width + kv_head * geometry.head_dim;
                let k_head = &key_cache[k_start..k_start + geometry.head_dim];
                scores.push(
                    q_head
                        .iter()
                        .zip(k_head)
                        .map(|(q_value, k_value)| q_value * k_value)
                        .sum::<f32>()
                        * scale,
                );
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0;
            for score in &mut scores {
                *score = (*score - maximum).exp();
                denominator += *score;
            }
            for score in &mut scores {
                *score /= denominator;
            }
            let context_head = &mut context[q_start..q_start + geometry.head_dim];
            for (position, weight) in scores.into_iter().enumerate() {
                let v_start = position * geometry.kv_width + kv_head * geometry.head_dim;
                let v_head = &value_cache[v_start..v_start + geometry.head_dim];
                for (output, value) in context_head.iter_mut().zip(v_head) {
                    *output += weight * value;
                }
            }
        }
        context
    }

    fn attention_complete_mirror(
        q: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        geometry: GpuNativeAttentionGeometry,
        seq_len: usize,
        o_projection: &DenseWeight,
        residual: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        assert_eq!(o_projection.rows(), geometry.d_model);
        assert_eq!(o_projection.cols(), geometry.q_width);
        assert_eq!(residual.len(), geometry.d_model);
        let context = causal_attention_mirror(q, key_cache, value_cache, geometry, seq_len);
        let projected = o_projection.matvec(&context);
        let hidden = residual
            .iter()
            .zip(&projected)
            .map(|(saved, contribution)| saved + contribution)
            .collect();
        (context, projected, hidden)
    }

    fn test_scratch<B>(
        context_id: u64,
        scratch_id: u64,
        elements: usize,
        buffer: B,
    ) -> GpuNativeScratch<B> {
        GpuNativeScratch::from_buffer(
            context_id,
            scratch_id,
            GpuNativeScratchLayout::try_new(elements).unwrap(),
            buffer,
        )
    }

    fn test_weight<B>(
        weight_id: u64,
        key: &str,
        layout: GpuNativeDenseWeightLayout,
        buffer: B,
    ) -> GpuNativeDenseWeight<B> {
        let plan = GpuNativeDenseWeightPlan::try_new(layout, &wgpu::Limits::default()).unwrap();
        assert_eq!(plan.chunks.len(), 1);
        GpuNativeDenseWeight {
            weight_id,
            key: GpuNativeDenseWeightKey::try_new(key).unwrap(),
            layout,
            chunks: vec![GpuNativeDenseWeightChunk {
                plan: plan.chunks[0],
                buffer,
            }],
        }
    }

    fn insert_test_f32_weight(
        registry: &mut GpuNativeDenseWeightRegistry<()>,
        weight_id: u64,
        key: &str,
        rows: usize,
        cols: usize,
    ) -> GpuNativeDenseWeightHandle {
        let bytes = rows
            .checked_mul(cols)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .unwrap();
        let layout =
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, rows, cols, bytes)
                .unwrap();
        registry
            .insert(test_weight(weight_id, key, layout, ()))
            .unwrap()
    }

    #[test]
    fn f32_persistent_weight_layout_charges_exact_bytes() {
        let weight = DenseWeight::from_f32(vec![0.0; 15], 3, 5);
        let layout = GpuNativeDenseWeightLayout::from_weight(&weight).unwrap();
        assert_eq!(layout.kind(), GpuNativeDenseWeightKind::F32);
        assert_eq!((layout.rows(), layout.cols()), (3, 5));
        assert_eq!(layout.payload_bytes(), 60);
        assert_eq!(layout.allocation_bytes(), 60);
    }

    #[test]
    fn q8_persistent_weight_layout_preserves_flat_blocks_and_wgpu_padding() {
        // 3x35 deliberately makes rows cross block boundaries and leaves a
        // final partial block: ceil(105/32) * 34 = 136 bytes.
        let values = (0..105).map(|i| i as f32 - 52.0).collect::<Vec<_>>();
        let weight = DenseWeight::from_q8_0_bytes(q8_bytes(&values), 3, 35).unwrap();
        let layout = GpuNativeDenseWeightLayout::from_weight(&weight).unwrap();
        assert_eq!(layout.kind(), GpuNativeDenseWeightKind::Q8_0);
        assert_eq!((layout.rows(), layout.cols()), (3, 35));
        assert_eq!(layout.payload_bytes(), 136);
        assert_eq!(layout.allocation_bytes(), 136);

        let one_block = GpuNativeDenseWeightLayout::try_new(
            GpuNativeDenseWeightKind::Q8_0,
            1,
            32,
            Q8_0_BLOCK_BYTES,
        )
        .unwrap();
        assert_eq!(one_block.payload_bytes(), 34);
        assert_eq!(one_block.allocation_bytes(), 36);
    }

    #[test]
    fn qwen_dense_weight_plans_fit_physical_storage_limits_without_allocating_payloads() {
        const ROWS: usize = 151_936;
        const COLS: usize = 2_048;
        const STORAGE_LIMIT: u64 = 128 * 1024 * 1024;
        const BUFFER_LIMIT: u64 = 256 * 1024 * 1024;
        let limits = wgpu::Limits {
            max_buffer_size: BUFFER_LIMIT,
            max_storage_buffer_binding_size: STORAGE_LIMIT as u32,
            ..wgpu::Limits::default()
        };

        let elements = ROWS * COLS;
        let f32_layout = GpuNativeDenseWeightLayout::try_new(
            GpuNativeDenseWeightKind::F32,
            ROWS,
            COLS,
            elements * std::mem::size_of::<f32>(),
        )
        .unwrap();
        let f32_plan = GpuNativeDenseWeightPlan::try_new(f32_layout, &limits).unwrap();
        assert_eq!(f32_plan.chunks.len(), 10);
        assert_eq!(f32_plan.chunks[0].row_count, 16_384);
        assert_eq!(f32_plan.chunks.last().unwrap().row_count, 4_480);
        assert!(f32_plan
            .chunks
            .iter()
            .all(|chunk| chunk.allocation_bytes <= STORAGE_LIMIT));
        assert_eq!(
            f32_plan
                .chunks
                .iter()
                .map(|chunk| chunk.allocation_bytes)
                .max(),
            Some(STORAGE_LIMIT)
        );
        for (index, chunk) in f32_plan.chunks.iter().enumerate() {
            super::super::validate_startup_buffer(
                &format!("test_qwen_f32_chunk_{index}"),
                chunk.allocation_bytes,
                GpuNativeDenseWeightLayout::usage(),
                &limits,
            )
            .unwrap();
        }

        let q8_bytes = elements.div_ceil(Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES;
        let q8_layout = GpuNativeDenseWeightLayout::try_new(
            GpuNativeDenseWeightKind::Q8_0,
            ROWS,
            COLS,
            q8_bytes,
        )
        .unwrap();
        let q8_plan = GpuNativeDenseWeightPlan::try_new(q8_layout, &limits).unwrap();
        assert_eq!(q8_plan.chunks.len(), 3);
        assert_eq!(q8_plan.chunks[0].row_count, 61_680);
        assert_eq!(q8_plan.chunks.last().unwrap().row_count, 28_576);
        assert!(q8_plan
            .chunks
            .iter()
            .all(|chunk| chunk.allocation_bytes <= STORAGE_LIMIT));
        assert_eq!(
            q8_plan
                .chunks
                .iter()
                .map(|chunk| chunk.allocation_bytes)
                .max(),
            Some(134_215_680)
        );
        for (index, chunk) in q8_plan.chunks.iter().enumerate() {
            super::super::validate_startup_buffer(
                &format!("test_qwen_q8_chunk_{index}"),
                chunk.allocation_bytes,
                GpuNativeDenseWeightLayout::usage(),
                &limits,
            )
            .unwrap();
        }
    }

    #[test]
    fn q8_row_crossing_chunks_preserve_metadata_blocks_gemv_and_embedding() {
        let rows = 3;
        let cols = 35;
        let values = (0..rows * cols)
            .map(|i| ((i * 17 % 43) as f32 - 21.0) / 7.0)
            .collect::<Vec<_>>();
        let source = q8_bytes(&values);
        let weight = DenseWeight::from_q8_0_bytes(source.clone(), rows, cols).unwrap();
        let layout = GpuNativeDenseWeightLayout::from_weight(&weight).unwrap();
        let limits = wgpu::Limits {
            max_buffer_size: 68,
            max_storage_buffer_binding_size: 68,
            ..wgpu::Limits::default()
        };
        let plan = GpuNativeDenseWeightPlan::try_new(layout, &limits).unwrap();
        assert_eq!(plan.chunks.len(), 3);
        assert_eq!(plan.physical_allocation_bytes, 204);
        assert_eq!(
            plan.chunks,
            vec![
                GpuNativeDenseWeightChunkPlan {
                    row_start: 0,
                    row_count: 1,
                    first_block: 0,
                    payload_offset_bytes: 0,
                    payload_bytes: 68,
                    allocation_bytes: 68,
                },
                GpuNativeDenseWeightChunkPlan {
                    row_start: 1,
                    row_count: 1,
                    first_block: 1,
                    payload_offset_bytes: 34,
                    payload_bytes: 68,
                    allocation_bytes: 68,
                },
                GpuNativeDenseWeightChunkPlan {
                    row_start: 2,
                    row_count: 1,
                    first_block: 2,
                    payload_offset_bytes: 68,
                    payload_bytes: 68,
                    allocation_bytes: 68,
                },
            ]
        );

        let chunk_bytes = plan
            .chunks
            .iter()
            .map(|chunk| {
                &source[chunk.payload_offset_bytes
                    ..chunk.payload_offset_bytes + chunk.payload_bytes as usize]
            })
            .collect::<Vec<_>>();
        assert_eq!(&chunk_bytes[0][34..68], &chunk_bytes[1][0..34]);
        assert_eq!(&chunk_bytes[1][34..68], &chunk_bytes[2][0..34]);

        for (chunk, bytes) in plan.chunks.iter().copied().zip(&chunk_bytes) {
            for row in chunk.row_start..chunk.row_end() {
                for col in 0..cols {
                    let flat = row * cols + col;
                    assert_eq!(
                        read_q8_chunk_mirror(bytes, chunk, flat),
                        read_q8_mirror(&source, flat)
                    );
                }
            }
        }

        let input = (0..cols)
            .map(|i| ((i * 11 % 19) as f32 - 9.0) / 5.0)
            .collect::<Vec<_>>();
        let expected_gemv = weight.matvec(&input);
        let mut chunked_gemv = vec![0.0; rows];
        for (chunk, bytes) in plan.chunks.iter().copied().zip(&chunk_bytes) {
            for row in chunk.row_start..chunk.row_end() {
                for col in 0..cols {
                    chunked_gemv[row] +=
                        read_q8_chunk_mirror(bytes, chunk, row * cols + col) * input[col];
                }
            }
        }
        assert_close(&chunked_gemv, &expected_gemv, 1e-5);

        for token in 0..rows {
            let (chunk, bytes) = plan
                .chunks
                .iter()
                .copied()
                .zip(&chunk_bytes)
                .find(|(chunk, _)| chunk.contains_row(token))
                .unwrap();
            let actual = (0..cols)
                .map(|col| read_q8_chunk_mirror(bytes, chunk, token * cols + col))
                .collect::<Vec<_>>();
            let mut expected = Vec::new();
            weight.row_dequant_into(token, &mut expected);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn dense_weight_planner_fails_when_one_complete_row_cannot_fit() {
        let layout =
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, 2, 4, 32).unwrap();
        let limits = wgpu::Limits {
            max_buffer_size: 15,
            max_storage_buffer_binding_size: 15,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            GpuNativeDenseWeightPlan::try_new(layout, &limits),
            Err(GpuNativeBootstrapError::DenseWeightRowExceedsDeviceLimit {
                kind: GpuNativeDenseWeightKind::F32,
                cols: 4,
                required: 16,
                maximum: 15,
            })
        );
    }

    #[test]
    fn dense_weight_layout_rejects_empty_malformed_and_overflowing_shapes() {
        assert_eq!(
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, 0, 4, 0),
            Err(GpuNativeBootstrapError::InvalidDenseWeightShape { rows: 0, cols: 4 })
        );
        assert_eq!(
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::Q8_0, 1, 32, 33),
            Err(GpuNativeBootstrapError::DenseWeightByteLength {
                kind: GpuNativeDenseWeightKind::Q8_0,
                rows: 1,
                cols: 32,
                expected: 34,
                actual: 33,
            })
        );
        assert_eq!(
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, usize::MAX, 2, 0,),
            Err(GpuNativeBootstrapError::DenseWeightShapeOverflow {
                rows: usize::MAX,
                cols: 2,
            })
        );
        if usize::BITS > u32::BITS {
            let rows = u32::MAX as usize + 1;
            assert_eq!(
                GpuNativeDenseWeightLayout::try_new(
                    GpuNativeDenseWeightKind::F32,
                    rows,
                    1,
                    rows * std::mem::size_of::<f32>(),
                ),
                Err(GpuNativeBootstrapError::DenseWeightDimensionTooLarge { rows, cols: 1 })
            );
        }
    }

    #[test]
    fn rms_norm_mirror_is_exactly_the_cpu_reference_for_required_widths() {
        for width in [1usize, 7, 65, 2_048] {
            let values = (0..width)
                .map(|index| ((index * 17 % 41) as f32 - 20.0) / 9.0)
                .collect::<Vec<_>>();
            let weight = (0..width)
                .map(|index| 0.75 + (index * 11 % 13) as f32 / 20.0)
                .collect::<Vec<_>>();
            let epsilon = 1e-6;
            let expected =
                crate::transformer::RmsNorm::new(weight.clone(), epsilon).forward(&values);
            assert_eq!(
                rms_norm_mirror(&values, &weight, epsilon, 1, width),
                expected,
                "width={width} must preserve the scalar f32 accumulation contract"
            );
        }

        let zero = vec![0.0; 7];
        assert_eq!(rms_norm_mirror(&zero, &[1.0; 7], 1e-6, 1, 7), zero);
    }

    #[test]
    fn grouped_rms_norm_matches_per_head_cpu_reference_and_isolates_boundaries() {
        let groups = 4;
        let group_width = 7;
        let epsilon = 1e-5;
        let weight = (0..group_width)
            .map(|index| 0.5 + index as f32 / 7.0)
            .collect::<Vec<_>>();
        let values = (0..groups * group_width)
            .map(|index| ((index * 19 % 37) as f32 - 18.0) / 5.0)
            .collect::<Vec<_>>();
        let actual = rms_norm_mirror(&values, &weight, epsilon, groups, group_width);
        let norm = crate::transformer::RmsNorm::new(weight.clone(), epsilon);
        let expected = values
            .chunks_exact(group_width)
            .flat_map(|group| norm.forward(group))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);

        let mut changed_last_group = values.clone();
        changed_last_group[3 * group_width..].fill(10_000.0);
        let changed = rms_norm_mirror(&changed_last_group, &weight, epsilon, groups, group_width);
        assert_eq!(
            &actual[..3 * group_width],
            &changed[..3 * group_width],
            "one Q/K head must not affect another head's RMS reduction"
        );
    }

    #[test]
    fn rms_norm_geometry_fails_closed_for_every_invalid_shape() {
        assert_eq!(
            validate_rms_norm_weight_width(0),
            Err(GpuNativeBootstrapError::InvalidRmsNormWeightWidth { width: 0 })
        );
        assert_eq!(validate_rms_norm_weight_width(2_048), Ok(()));
        assert_eq!(
            GpuNativeRmsNormGeometry::try_new(0, 7, 0, 7),
            Err(GpuNativeBootstrapError::InvalidRmsNormGroups { groups: 0 })
        );
        assert_eq!(
            GpuNativeRmsNormGeometry::try_new(3, 0, 0, 0),
            Err(GpuNativeBootstrapError::InvalidRmsNormGroupWidth { group_width: 0 })
        );
        assert_eq!(
            GpuNativeRmsNormGeometry::try_new(usize::MAX, 2, 0, 2),
            Err(GpuNativeBootstrapError::RmsNormGeometryOverflow {
                groups: usize::MAX,
                group_width: 2,
            })
        );
        if usize::BITS > u32::BITS {
            let groups = u32::MAX as usize + 1;
            assert_eq!(
                GpuNativeRmsNormGeometry::try_new(groups, 1, groups, 1),
                Err(GpuNativeBootstrapError::RmsNormGeometryOverflow {
                    groups,
                    group_width: 1,
                })
            );
        }
        assert_eq!(
            GpuNativeRmsNormGeometry::try_new(3, 7, 20, 7),
            Err(GpuNativeBootstrapError::RmsNormScratchGeometry {
                expected: 21,
                actual: 20,
            })
        );
        assert_eq!(
            GpuNativeRmsNormGeometry::try_new(3, 7, 21, 8),
            Err(GpuNativeBootstrapError::RmsNormWeightWidth {
                expected: 7,
                actual: 8,
            })
        );
        let geometry = GpuNativeRmsNormGeometry::try_new(4, 7, 28, 7).unwrap();
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 3,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            geometry.checked_workgroups(&limits),
            Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
                workgroups: 4,
                maximum: 3,
            })
        );
    }

    #[test]
    fn rms_norm_epsilon_rejects_non_finite_and_negative_values() {
        for epsilon in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1e-6] {
            assert_eq!(
                validate_rms_norm_epsilon(epsilon),
                Err(GpuNativeBootstrapError::InvalidRmsNormEpsilon {
                    epsilon_bits: epsilon.to_bits(),
                })
            );
        }

        assert_eq!(validate_rms_norm_epsilon(0.0), Ok(()));
        assert_eq!(validate_rms_norm_epsilon(1e-6), Ok(()));
    }

    #[test]
    fn attention_geometry_accepts_qwen_gqa_and_non_power_of_two_head_counts() {
        let geometry = GpuNativeAttentionGeometry::try_new(12, 6, 2, 4, 4).unwrap();
        assert_eq!(geometry.d_model(), 12);
        assert_eq!(geometry.num_heads(), 6);
        assert_eq!(geometry.num_kv_heads(), 2);
        assert_eq!(geometry.head_dim(), 4);
        assert_eq!(geometry.rope_dim(), 4);
        assert_eq!(geometry.q_width(), 24);
        assert_eq!(geometry.kv_width(), 8);

        assert_eq!(
            GpuNativeAttentionGeometry::try_new(12, 0, 2, 4, 4),
            Err(GpuNativeBootstrapError::InvalidAttentionHeadCount {
                tensor: GpuNativeAttentionTensor::Query,
                heads: 0,
            })
        );
        assert_eq!(
            GpuNativeAttentionGeometry::try_new(12, 6, 0, 4, 4),
            Err(GpuNativeBootstrapError::InvalidAttentionHeadCount {
                tensor: GpuNativeAttentionTensor::Key,
                heads: 0,
            })
        );
        assert!(matches!(
            GpuNativeAttentionGeometry::try_new(12, 6, 4, 4, 4),
            Err(GpuNativeBootstrapError::InvalidAttentionHeadGeometry { .. })
        ));
        assert_eq!(
            GpuNativeAttentionGeometry::try_new(12, 6, 2, 4, 3),
            Err(GpuNativeBootstrapError::OddRopeDimension { rope_dim: 3 })
        );
        assert_eq!(
            GpuNativeAttentionGeometry::try_new(12, 6, 2, 4, 6),
            Err(GpuNativeBootstrapError::InvalidRopeDimension {
                rope_dim: 6,
                head_dim: 4,
            })
        );
        assert!(matches!(
            GpuNativeAttentionGeometry::try_new(12, u32::MAX as usize, 1, 2, 2),
            Err(GpuNativeBootstrapError::AttentionGeometryOverflow { .. })
        ));
    }

    #[test]
    fn attention_geometry_keeps_q_width_distinct_from_d_model() {
        let geometry = GpuNativeAttentionGeometry::try_new(6, 4, 2, 2, 2).unwrap();
        assert_eq!(geometry.d_model(), 6);
        assert_eq!(geometry.q_width(), 8);
        assert_eq!(geometry.kv_width(), 4);
        assert_ne!(geometry.q_width(), geometry.d_model());
    }

    #[test]
    fn causal_attention_dispatch_fails_closed_for_device_limits() {
        let geometry = GpuNativeAttentionGeometry::try_new(6, 4, 2, 2, 2).unwrap();
        assert_eq!(
            validate_causal_attention_dispatch(geometry, &wgpu::Limits::default()),
            Ok(4)
        );
        let narrow_dispatch = wgpu::Limits {
            max_compute_workgroups_per_dimension: 3,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            validate_causal_attention_dispatch(geometry, &narrow_dispatch),
            Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
                workgroups: 4,
                maximum: 3,
            })
        );
        let narrow_workgroup = wgpu::Limits {
            max_compute_workgroup_size_x: 16,
            max_compute_invocations_per_workgroup: 16,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            validate_causal_attention_dispatch(geometry, &narrow_workgroup),
            Err(GpuNativeBootstrapError::AttentionWorkgroupUnsupported {
                required: 32,
                max_size_x: 16,
                max_invocations: 16,
            })
        );
    }

    #[test]
    fn attention_plan_validates_layer_and_output_projection_shape() {
        let context_id = 41;
        let geometry = GpuNativeAttentionGeometry::try_new(6, 4, 2, 2, 2).unwrap();
        let mut registry = GpuNativeDenseWeightRegistry::new(context_id);
        let q_projection = insert_test_f32_weight(
            &mut registry,
            1,
            "layer.1.attention.q",
            geometry.q_width,
            geometry.d_model,
        );
        let k_projection = insert_test_f32_weight(
            &mut registry,
            2,
            "layer.1.attention.k",
            geometry.kv_width,
            geometry.d_model,
        );
        let v_projection = insert_test_f32_weight(
            &mut registry,
            3,
            "layer.1.attention.v",
            geometry.kv_width,
            geometry.d_model,
        );
        let o_projection = insert_test_f32_weight(
            &mut registry,
            4,
            "layer.1.attention.o",
            geometry.d_model,
            geometry.q_width,
        );
        let rope_dense = insert_test_f32_weight(
            &mut registry,
            5,
            "layer.1.attention.rope",
            1,
            geometry.rope_dim / 2,
        );
        let plan = GpuNativeAttentionPlan {
            context_id,
            layer_index: 1,
            geometry,
            q_projection,
            k_projection,
            v_projection,
            o_projection,
            q_norm: None,
            k_norm: None,
            rope: GpuNativeRopeHandle {
                dense: rope_dense,
                rope_dim: geometry.rope_dim,
                attention_factor_bits: 1.0f32.to_bits(),
            },
        };
        assert_eq!(plan.layer_index(), 1);
        assert_eq!(
            validate_attention_plan_with_registry(context_id, geometry.d_model, &registry, &plan),
            Ok(())
        );

        let transposed_o = insert_test_f32_weight(
            &mut registry,
            6,
            "layer.1.attention.o_transposed",
            geometry.q_width,
            geometry.d_model,
        );
        let mut wrong = plan.clone();
        wrong.o_projection = transposed_o;
        assert_eq!(
            validate_attention_plan_with_registry(context_id, geometry.d_model, &registry, &wrong),
            Err(GpuNativeBootstrapError::AttentionProjectionShape {
                tensor: GpuNativeAttentionTensor::Output,
                expected_rows: geometry.d_model,
                expected_cols: geometry.q_width,
                actual_rows: geometry.q_width,
                actual_cols: geometry.d_model,
            })
        );

        let square_o = insert_test_f32_weight(
            &mut registry,
            7,
            "layer.1.attention.o_square",
            geometry.d_model,
            geometry.d_model,
        );
        wrong.o_projection = square_o;
        assert_eq!(
            validate_attention_plan_with_registry(context_id, geometry.d_model, &registry, &wrong),
            Err(GpuNativeBootstrapError::AttentionProjectionShape {
                tensor: GpuNativeAttentionTensor::Output,
                expected_rows: geometry.d_model,
                expected_cols: geometry.q_width,
                actual_rows: geometry.d_model,
                actual_cols: geometry.d_model,
            })
        );

        wrong = plan.clone();
        wrong.o_projection.context_id += 1;
        assert_eq!(
            validate_attention_plan_with_registry(context_id, geometry.d_model, &registry, &wrong),
            Err(GpuNativeBootstrapError::ForeignDenseWeightHandle)
        );
        wrong = plan.clone();
        wrong.o_projection.weight_id += 1;
        assert!(matches!(
            validate_attention_plan_with_registry(context_id, geometry.d_model, &registry, &wrong),
            Err(GpuNativeBootstrapError::StaleDenseWeightHandle { .. })
        ));
    }

    #[test]
    fn layer_bound_kv_and_causal_sequence_lengths_fail_closed() {
        let geometry = GpuNativeAttentionGeometry::try_new(6, 4, 2, 2, 2).unwrap();
        let layout =
            GpuNativeKvLayout::try_new(2, 4, geometry.kv_width, &wgpu::Limits::default()).unwrap();
        let kv = GpuNativeKvState::from_layers(
            7,
            1,
            layout,
            vec![
                GpuNativeKvLayer { key: (), value: () },
                GpuNativeKvLayer { key: (), value: () },
            ],
        );
        assert_eq!(validate_attention_kv_state(7, geometry, &kv, 1, 0), Ok(1));
        assert_eq!(validate_attention_kv_state(7, geometry, &kv, 1, 2), Ok(3));
        assert_eq!(validate_attention_kv_state(7, geometry, &kv, 1, 3), Ok(4));
        assert_eq!(
            validate_attention_kv_state(7, geometry, &kv, 2, 0),
            Err(GpuNativeBootstrapError::AttentionPlanLayerOutOfRange {
                layer_index: 2,
                num_layers: 2,
            })
        );
        assert_eq!(
            validate_attention_kv_state(7, geometry, &kv, 1, 4),
            Err(GpuNativeBootstrapError::InvalidAttentionSequenceLength {
                seq_len: 5,
                max_seq_len: 4,
            })
        );
        assert_eq!(
            validate_attention_kv_state(7, geometry, &kv, 1, usize::MAX),
            Err(GpuNativeBootstrapError::AttentionSequenceLengthOverflow {
                position: usize::MAX,
            })
        );
    }

    #[test]
    fn qk_norm_groups_are_independent_for_gqa_geometry() {
        let geometry = GpuNativeAttentionGeometry::try_new(12, 6, 2, 4, 4).unwrap();
        let gain = [0.7, 1.1, 0.9, 1.3];
        let epsilon = 1e-6;
        let q = (0..geometry.q_width)
            .map(|index| (index as f32 - 11.0) / 3.0)
            .collect::<Vec<_>>();
        let k = (0..geometry.kv_width)
            .map(|index| (index as f32 - 3.0) / 2.0)
            .collect::<Vec<_>>();
        let q_norm = rms_norm_mirror(&q, &gain, epsilon, geometry.num_heads, geometry.head_dim);
        let k_norm = rms_norm_mirror(&k, &gain, epsilon, geometry.num_kv_heads, geometry.head_dim);
        let cpu_norm = crate::transformer::RmsNorm::new(gain.to_vec(), epsilon);
        let expected_q = q
            .chunks_exact(geometry.head_dim)
            .flat_map(|head| cpu_norm.forward(head))
            .collect::<Vec<_>>();
        let expected_k = k
            .chunks_exact(geometry.head_dim)
            .flat_map(|head| cpu_norm.forward(head))
            .collect::<Vec<_>>();
        assert_eq!(q_norm, expected_q);
        assert_eq!(k_norm, expected_k);

        let mut changed_q = q.clone();
        changed_q[5 * geometry.head_dim..].fill(10_000.0);
        let changed_q_norm = rms_norm_mirror(
            &changed_q,
            &gain,
            epsilon,
            geometry.num_heads,
            geometry.head_dim,
        );
        assert_eq!(
            &q_norm[..5 * geometry.head_dim],
            &changed_q_norm[..5 * geometry.head_dim]
        );
    }

    #[test]
    fn rope_mirror_matches_cpu_pairing_positions_and_partial_tail() {
        const GROUPS: usize = 3;
        const HEAD_DIM: usize = 6;
        const ROPE_DIM: usize = 4;
        let inverse_frequencies = [1.0, 0.01];
        let values = (0..GROUPS * HEAD_DIM)
            .map(|index| (index as f32 - 7.0) / 4.0)
            .collect::<Vec<_>>();
        assert_eq!(
            rope_mirror(
                &values,
                GROUPS,
                HEAD_DIM,
                ROPE_DIM,
                0,
                &inverse_frequencies,
                1.0,
            ),
            values
        );

        let actual = rope_mirror(
            &values,
            GROUPS,
            HEAD_DIM,
            ROPE_DIM,
            7,
            &inverse_frequencies,
            1.0,
        );
        let mut expected = values.clone();
        for head in expected.chunks_exact_mut(HEAD_DIM) {
            crate::transformer::apply_rope_inplace(&mut head[..ROPE_DIM], 7, 10_000.0);
        }
        assert_close(&actual, &expected, 1e-6);
        for head in 0..GROUPS {
            let start = head * HEAD_DIM;
            assert_eq!(
                &actual[start + ROPE_DIM..start + HEAD_DIM],
                &values[start + ROPE_DIM..start + HEAD_DIM]
            );
        }

        let one_head = rope_mirror(&[1.0, 2.0, 3.0, 4.0], 1, 4, 4, 1, &inverse_frequencies, 1.0);
        let mut cpu_pairing = vec![1.0, 2.0, 3.0, 4.0];
        crate::transformer::apply_rope_inplace(&mut cpu_pairing, 1, 10_000.0);
        assert_close(&one_head, &cpu_pairing, 1e-6);
    }

    #[test]
    fn rope_parameters_and_typed_handle_fail_closed() {
        let layout = GpuNativeRopeLayout::try_new(4, 4).unwrap();
        assert_close(
            &standard_rope_inverse_frequencies(4, 10_000.0).unwrap(),
            &[1.0, 0.01],
            1e-7,
        );
        for invalid_base in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                standard_rope_inverse_frequencies(4, invalid_base),
                Err(GpuNativeBootstrapError::InvalidRopeBase {
                    base_bits: invalid_base.to_bits(),
                })
            );
        }
        assert_eq!(validate_rope_parameters(layout, &[1.0, 0.01], 1.0), Ok(()));
        assert_eq!(
            validate_rope_parameters(layout, &[1.0], 1.0),
            Err(GpuNativeBootstrapError::RopeParameterWidth {
                expected: 2,
                actual: 1,
            })
        );
        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                validate_rope_parameters(layout, &[1.0, invalid], 1.0),
                Err(GpuNativeBootstrapError::InvalidRopeInverseFrequency {
                    index: 1,
                    value_bits: invalid.to_bits(),
                })
            );
        }
        assert_eq!(
            validate_rope_parameters(layout, &[1.0, 0.01], f32::NAN),
            Err(GpuNativeBootstrapError::InvalidRopeAttentionFactor {
                factor_bits: f32::NAN.to_bits(),
            })
        );

        let dense_layout =
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, 1, 2, 8).unwrap();
        let mut registry = GpuNativeDenseWeightRegistry::new(41);
        let dense = registry
            .insert(test_weight(7, "model.rope", dense_layout, ()))
            .unwrap();
        let handle = GpuNativeRopeHandle {
            dense,
            rope_dim: 4,
            attention_factor_bits: 1.0f32.to_bits(),
        };
        assert_eq!(
            validate_rope_handle_with_registry(41, &registry, &handle, 4),
            Ok(())
        );
        assert_eq!(
            validate_rope_handle_with_registry(42, &registry, &handle, 4),
            Err(GpuNativeBootstrapError::ForeignRopeHandle)
        );
        let mut stale = handle.clone();
        stale.dense.weight_id += 1;
        assert!(matches!(
            validate_rope_handle_with_registry(41, &registry, &stale, 4),
            Err(GpuNativeBootstrapError::StaleRopeHandle { .. })
        ));
    }

    #[test]
    fn kv_layout_checks_offsets_capacity_overflow_and_binding_limits() {
        let layout = GpuNativeKvLayout::try_new(3, 5, 8, &wgpu::Limits::default()).unwrap();
        assert_eq!(layout.num_layers(), 3);
        assert_eq!(layout.max_seq_len(), 5);
        assert_eq!(layout.kv_width(), 8);
        assert_eq!(layout.layer_bytes(), 160);
        assert_eq!(layout.total_bytes(), 960);
        assert_eq!(layout.element_offset(2, 4), Ok(32));
        assert_eq!(
            layout.element_offset(3, 0),
            Err(GpuNativeBootstrapError::InvalidKvLayer {
                layer: 3,
                num_layers: 3,
            })
        );
        assert_eq!(
            layout.element_offset(0, 5),
            Err(GpuNativeBootstrapError::InvalidKvPosition {
                position: 5,
                max_seq_len: 5,
            })
        );
        assert!(matches!(
            GpuNativeKvLayout::try_new(1, usize::MAX, 8, &wgpu::Limits::default()),
            Err(GpuNativeBootstrapError::KvCapacityOverflow { .. })
        ));
        let limits = wgpu::Limits {
            max_buffer_size: 128,
            max_storage_buffer_binding_size: 128,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            GpuNativeKvLayout::try_new(1, 5, 8, &limits),
            Err(GpuNativeBootstrapError::KvBufferLimit {
                required: 160,
                max_buffer_size: 128,
                max_storage_binding_size: 128,
            })
        );
        assert!(!GpuNativeKvLayout::usage()
            .intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));
        assert!(!GpuNativeKvLayout::usage().contains(wgpu::BufferUsages::COPY_SRC));
    }

    #[test]
    fn attention_scratch_and_kv_ownership_and_widths_fail_closed() {
        let geometry = GpuNativeAttentionGeometry::try_new(12, 6, 2, 4, 4).unwrap();
        let scratch = GpuNativeAttentionScratch::from_scratch(
            7,
            geometry,
            test_scratch(7, 1, geometry.q_width, ()),
            test_scratch(7, 2, geometry.kv_width, ()),
            test_scratch(7, 3, geometry.kv_width, ()),
            test_scratch(7, 4, geometry.q_width, ()),
            test_scratch(7, 5, geometry.d_model, ()),
        );
        assert_eq!(validate_attention_scratch(7, geometry, &scratch), Ok(()));
        assert_eq!(
            validate_attention_scratch(8, geometry, &scratch),
            Err(GpuNativeBootstrapError::ForeignAttentionScratch)
        );
        let wrong_q = GpuNativeAttentionScratch::from_scratch(
            7,
            geometry,
            test_scratch(7, 6, geometry.q_width - 1, ()),
            test_scratch(7, 7, geometry.kv_width, ()),
            test_scratch(7, 8, geometry.kv_width, ()),
            test_scratch(7, 9, geometry.q_width, ()),
            test_scratch(7, 10, geometry.d_model, ()),
        );
        assert_eq!(
            validate_attention_scratch(7, geometry, &wrong_q),
            Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor: GpuNativeAttentionTensor::Query,
                expected: geometry.q_width,
                actual: geometry.q_width - 1,
            })
        );
        let wrong_k = GpuNativeAttentionScratch::from_scratch(
            7,
            geometry,
            test_scratch(7, 11, geometry.q_width, ()),
            test_scratch(7, 12, geometry.kv_width - 1, ()),
            test_scratch(7, 13, geometry.kv_width, ()),
            test_scratch(7, 14, geometry.q_width, ()),
            test_scratch(7, 15, geometry.d_model, ()),
        );
        assert_eq!(
            validate_attention_scratch(7, geometry, &wrong_k),
            Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor: GpuNativeAttentionTensor::Key,
                expected: geometry.kv_width,
                actual: geometry.kv_width - 1,
            })
        );
        let wrong_v = GpuNativeAttentionScratch::from_scratch(
            7,
            geometry,
            test_scratch(7, 16, geometry.q_width, ()),
            test_scratch(7, 17, geometry.kv_width, ()),
            test_scratch(7, 18, geometry.kv_width - 1, ()),
            test_scratch(7, 19, geometry.q_width, ()),
            test_scratch(7, 20, geometry.d_model, ()),
        );
        assert_eq!(
            validate_attention_scratch(7, geometry, &wrong_v),
            Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor: GpuNativeAttentionTensor::Value,
                expected: geometry.kv_width,
                actual: geometry.kv_width - 1,
            })
        );
        let wrong_context = GpuNativeAttentionScratch::from_scratch(
            7,
            geometry,
            test_scratch(7, 21, geometry.q_width, ()),
            test_scratch(7, 22, geometry.kv_width, ()),
            test_scratch(7, 23, geometry.kv_width, ()),
            test_scratch(7, 24, geometry.q_width - 1, ()),
            test_scratch(7, 25, geometry.d_model, ()),
        );
        assert_eq!(
            validate_attention_scratch(7, geometry, &wrong_context),
            Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor: GpuNativeAttentionTensor::Context,
                expected: geometry.q_width,
                actual: geometry.q_width - 1,
            })
        );
        let wrong_projected = GpuNativeAttentionScratch::from_scratch(
            7,
            geometry,
            test_scratch(7, 26, geometry.q_width, ()),
            test_scratch(7, 27, geometry.kv_width, ()),
            test_scratch(7, 28, geometry.kv_width, ()),
            test_scratch(7, 29, geometry.q_width, ()),
            test_scratch(7, 30, geometry.d_model - 1, ()),
        );
        assert_eq!(
            validate_attention_scratch(7, geometry, &wrong_projected),
            Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor: GpuNativeAttentionTensor::Output,
                expected: geometry.d_model,
                actual: geometry.d_model - 1,
            })
        );
        let aliased = GpuNativeAttentionScratch::from_scratch(
            7,
            geometry,
            test_scratch(7, 31, geometry.q_width, ()),
            test_scratch(7, 32, geometry.kv_width, ()),
            test_scratch(7, 33, geometry.kv_width, ()),
            test_scratch(7, 31, geometry.q_width, ()),
            test_scratch(7, 34, geometry.d_model, ()),
        );
        assert_eq!(
            validate_attention_scratch(7, geometry, &aliased),
            Err(GpuNativeBootstrapError::AliasedInputOutput)
        );

        let kv_layout = GpuNativeKvLayout::try_new(2, 4, 8, &wgpu::Limits::default()).unwrap();
        let kv = GpuNativeKvState::from_layers(
            7,
            1,
            kv_layout,
            vec![
                GpuNativeKvLayer { key: (), value: () },
                GpuNativeKvLayer { key: (), value: () },
            ],
        );
        assert_eq!(validate_kv_state(7, 8, &kv, 1, 3), Ok(()));
        assert_eq!(
            validate_kv_state(8, 8, &kv, 1, 3),
            Err(GpuNativeBootstrapError::ForeignKvState)
        );
        assert_eq!(
            validate_kv_state(7, 7, &kv, 1, 3),
            Err(GpuNativeBootstrapError::KvWidth {
                expected: 7,
                actual: 8,
            })
        );
        assert!(matches!(
            validate_kv_state(7, 8, &kv, 2, 3),
            Err(GpuNativeBootstrapError::InvalidKvLayer { .. })
        ));
        assert!(matches!(
            validate_kv_state(7, 8, &kv, 1, 4),
            Err(GpuNativeBootstrapError::InvalidKvPosition { .. })
        ));
    }

    #[test]
    fn hardware_independent_attention_prepare_chain_preserves_two_kv_positions() {
        let geometry = GpuNativeAttentionGeometry::try_new(6, 4, 2, 4, 4).unwrap();
        let q_projection = DenseWeight::from_f32(
            (0..geometry.q_width * geometry.d_model)
                .map(|index| ((index * 11 % 37) as f32 - 18.0) / 19.0)
                .collect(),
            geometry.q_width,
            geometry.d_model,
        );
        let k_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * geometry.d_model)
                .map(|index| ((index * 13 % 41) as f32 - 20.0) / 17.0)
                .collect(),
            geometry.kv_width,
            geometry.d_model,
        );
        let v_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * geometry.d_model)
                .map(|index| ((index * 17 % 43) as f32 - 21.0) / 23.0)
                .collect(),
            geometry.kv_width,
            geometry.d_model,
        );
        let q_gain = [0.8, 1.1, 0.9, 1.2];
        let k_gain = [1.3, 0.7, 1.0, 0.85];
        let epsilon = 1e-6;
        let inverse_frequencies = [1.0, 0.01];
        let kv_layout =
            GpuNativeKvLayout::try_new(1, 4, geometry.kv_width, &wgpu::Limits::default()).unwrap();
        let mut key_cache = vec![f32::NAN; kv_layout.layer_elements];
        let mut value_cache = vec![f32::NAN; kv_layout.layer_elements];
        let mut first_key = Vec::new();
        let mut first_value = Vec::new();

        for (position, hidden) in [
            (1usize, vec![0.5, -1.0, 1.5, -2.0, 2.5, -3.0]),
            (3usize, vec![-0.25, 0.75, -1.25, 1.75, -2.25, 2.75]),
        ] {
            let (q, k, v) = attention_prepare_mirror(
                &hidden,
                geometry,
                &q_projection,
                &k_projection,
                &v_projection,
                Some((&q_gain, epsilon)),
                Some((&k_gain, epsilon)),
                &inverse_frequencies,
                1.0,
                position,
            );
            assert_eq!(q.len(), geometry.q_width);
            assert_eq!(k.len(), geometry.kv_width);
            assert_eq!(v.len(), geometry.kv_width);
            assert!(q.iter().chain(&k).chain(&v).all(|value| value.is_finite()));
            let offset = kv_layout.element_offset(0, position).unwrap();
            key_cache[offset..offset + geometry.kv_width].copy_from_slice(&k);
            value_cache[offset..offset + geometry.kv_width].copy_from_slice(&v);
            if position == 1 {
                first_key = k;
                first_value = v;
            }
        }

        let first_offset = kv_layout.element_offset(0, 1).unwrap();
        assert_eq!(
            &key_cache[first_offset..first_offset + geometry.kv_width],
            first_key
        );
        assert_eq!(
            &value_cache[first_offset..first_offset + geometry.kv_width],
            first_value
        );
        assert!(key_cache[..geometry.kv_width]
            .iter()
            .all(|value| value.is_nan()));
        let unused_offset = kv_layout.element_offset(0, 2).unwrap();
        assert!(key_cache[unused_offset..unused_offset + geometry.kv_width]
            .iter()
            .all(|value| value.is_nan()));
    }

    #[test]
    fn causal_gqa_attention_o_projection_and_residual_match_cpu_mirror() {
        let geometry = GpuNativeAttentionGeometry::try_new(6, 4, 2, 2, 2).unwrap();
        let q = [0.8, -0.3, 1.1, 0.4, -0.7, 1.3, 0.2, -1.0];
        let key_cache = [
            0.2, -0.5, 1.1, 0.3, // position 0: KV heads 0, 1
            0.9, 0.1, -0.4, 0.8, // position 1
            -0.6, 1.2, 0.5, -0.9, // position 2
            10_000.0, -9_000.0, 8_000.0, -7_000.0, // future poison
        ];
        let value_cache = [
            0.5, -1.0, 1.5, 0.25, // position 0
            -0.75, 1.25, 0.4, -1.4, // position 1
            1.8, 0.6, -0.9, 1.1, // position 2
            50_000.0, -40_000.0, 30_000.0, -20_000.0, // future poison
        ];
        let o_projection = DenseWeight::from_f32(
            (0..geometry.d_model * geometry.q_width)
                .map(|index| ((index * 17 % 31) as f32 - 15.0) / 13.0)
                .collect(),
            geometry.d_model,
            geometry.q_width,
        );
        let residual = [0.25, -0.5, 0.75, -1.0, 1.25, -1.5];
        let residual_before = residual;
        let (context, projected, hidden) = attention_complete_mirror(
            &q,
            &key_cache,
            &value_cache,
            geometry,
            3,
            &o_projection,
            &residual,
        );
        assert_eq!(context.len(), geometry.q_width);
        assert_eq!(projected.len(), geometry.d_model);
        assert_eq!(hidden.len(), geometry.d_model);
        assert!(context
            .iter()
            .chain(&projected)
            .chain(&hidden)
            .all(|value| value.is_finite()));
        assert_eq!(residual, residual_before);
        for ((actual, saved), contribution) in hidden.iter().zip(residual).zip(&projected) {
            assert!((actual - (saved + contribution)).abs() < 1e-6);
        }

        let mut changed_future_keys = key_cache;
        let mut changed_future_values = value_cache;
        changed_future_keys[3 * geometry.kv_width..].fill(-123_456.0);
        changed_future_values[3 * geometry.kv_width..].fill(654_321.0);
        let future_poisoned = causal_attention_mirror(
            &q,
            &changed_future_keys,
            &changed_future_values,
            geometry,
            3,
        );
        assert_close(&context, &future_poisoned, 0.0);

        let uniform_head_zero = [
            (value_cache[0] + value_cache[4] + value_cache[8]) / 3.0,
            (value_cache[1] + value_cache[5] + value_cache[9]) / 3.0,
        ];
        assert!(context[..geometry.head_dim]
            .iter()
            .zip(uniform_head_zero)
            .any(|(actual, uniform)| (actual - uniform).abs() > 1e-3));
        // Heads 0 and 1 share KV head 0; heads 2 and 3 share KV head 1.
        assert_ne!(
            &context[..2 * geometry.head_dim],
            &context[2 * geometry.head_dim..]
        );
    }

    #[test]
    fn attention_prepare_allows_absent_qk_norm_as_a_clean_noop() {
        let geometry = GpuNativeAttentionGeometry::try_new(4, 2, 1, 4, 4).unwrap();
        let q_projection = DenseWeight::from_f32(
            (0..geometry.q_width * geometry.d_model)
                .map(|index| (index as f32 - 8.0) / 7.0)
                .collect(),
            geometry.q_width,
            geometry.d_model,
        );
        let k_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * geometry.d_model)
                .map(|index| (index as f32 - 5.0) / 9.0)
                .collect(),
            geometry.kv_width,
            geometry.d_model,
        );
        let v_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * geometry.d_model)
                .map(|index| (index as f32 - 3.0) / 11.0)
                .collect(),
            geometry.kv_width,
            geometry.d_model,
        );
        let hidden = [0.25, -0.5, 0.75, -1.0];
        let inverse_frequencies = [1.0, 0.01];
        let (q, k, v) = attention_prepare_mirror(
            &hidden,
            geometry,
            &q_projection,
            &k_projection,
            &v_projection,
            None,
            None,
            &inverse_frequencies,
            1.0,
            2,
        );
        assert_close(
            &q,
            &rope_mirror(
                &q_projection.matvec(&hidden),
                geometry.num_heads,
                geometry.head_dim,
                geometry.rope_dim,
                2,
                &inverse_frequencies,
                1.0,
            ),
            1e-6,
        );
        assert_close(
            &k,
            &rope_mirror(
                &k_projection.matvec(&hidden),
                geometry.num_kv_heads,
                geometry.head_dim,
                geometry.rope_dim,
                2,
                &inverse_frequencies,
                1.0,
            ),
            1e-6,
        );
        assert_eq!(v, v_projection.matvec(&hidden));
    }

    #[test]
    fn typed_rms_norm_handles_reuse_persistent_f32_registry_and_fail_closed() {
        let layout =
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, 1, 7, 28).unwrap();
        let mut registry = GpuNativeDenseWeightRegistry::new(41);
        let dense = registry
            .insert(test_weight(7, "layer.0.rms_attn", layout, ()))
            .unwrap();
        let handle = GpuNativeRmsNormHandle::from_dense(dense.clone());
        assert_eq!(handle.width(), 7);
        let first = registry.resolve_rms_norm(&handle).unwrap();
        let second = registry.resolve_rms_norm(&handle).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(registry.weights.len(), 1);

        let foreign_registry = GpuNativeDenseWeightRegistry::<()>::new(42);
        assert!(matches!(
            foreign_registry.resolve_rms_norm(&handle),
            Err(GpuNativeBootstrapError::ForeignRmsNormHandle)
        ));

        let mut stale = handle.clone();
        stale.dense.weight_id += 1;
        assert!(matches!(
            registry.resolve_rms_norm(&stale),
            Err(GpuNativeBootstrapError::StaleRmsNormHandle { .. })
        ));

        let mut wrong_width = handle.clone();
        wrong_width.width = 6;
        assert!(matches!(
            registry.resolve_rms_norm(&wrong_width),
            Err(GpuNativeBootstrapError::StaleRmsNormHandle { .. })
        ));
        assert_eq!(
            registry.handle_for(dense.key(), GpuNativeDenseWeightKind::F32, 1, 6,),
            Err(GpuNativeBootstrapError::DenseWeightShapeMismatch {
                key: "layer.0.rms_attn".to_string(),
                expected_rows: 1,
                expected_cols: 6,
                actual_rows: 1,
                actual_cols: 7,
            })
        );

        let q8_layout = GpuNativeDenseWeightLayout::try_new(
            GpuNativeDenseWeightKind::Q8_0,
            1,
            32,
            Q8_0_BLOCK_BYTES,
        )
        .unwrap();
        let q8_dense = registry
            .insert(test_weight(8, "invalid.q8_norm", q8_layout, ()))
            .unwrap();
        let q8_handle = GpuNativeRmsNormHandle::from_dense(q8_dense);
        assert!(matches!(
            registry.resolve_rms_norm(&q8_handle),
            Err(GpuNativeBootstrapError::StaleRmsNormHandle { .. })
        ));
    }

    #[test]
    fn new_state_operations_reject_foreign_request_ownership() {
        assert_eq!(validate_token_state_owner(7, 7), Ok(()));
        assert_eq!(validate_scratch_owner(7, 7), Ok(()));
        assert_eq!(
            validate_token_state_owner(7, 8),
            Err(GpuNativeBootstrapError::ForeignTokenState)
        );
        assert_eq!(
            validate_scratch_owner(7, 8),
            Err(GpuNativeBootstrapError::ForeignScratch)
        );
        assert_eq!(validate_residual_contribution_width(2_048, 2_048), Ok(()));
        assert_eq!(
            validate_residual_contribution_width(2_048, 1_024),
            Err(GpuNativeBootstrapError::ResidualContributionWidth {
                expected: 2_048,
                actual: 1_024,
            })
        );
    }

    #[test]
    fn hardware_independent_residual_chain_preserves_every_state_transition() {
        const WIDTH: usize = 7;
        let embedding = DenseWeight::from_f32(
            (0..3 * WIDTH)
                .map(|index| ((index * 13 % 29) as f32 - 14.0) / 6.0)
                .collect(),
            3,
            WIDTH,
        );
        let first_dense = DenseWeight::from_f32(
            (0..WIDTH * WIDTH)
                .map(|index| ((index * 17 % 31) as f32 - 15.0) / 23.0)
                .collect(),
            WIDTH,
            WIDTH,
        );
        let second_dense = DenseWeight::from_f32(
            (0..WIDTH * WIDTH)
                .map(|index| ((index * 19 % 37) as f32 - 18.0) / 29.0)
                .collect(),
            WIDTH,
            WIDTH,
        );
        let first_gain = (0..WIDTH)
            .map(|index| 0.7 + index as f32 / 20.0)
            .collect::<Vec<_>>();
        let second_gain = (0..WIDTH)
            .map(|index| 0.9 - index as f32 / 30.0)
            .collect::<Vec<_>>();
        let final_gain = (0..WIDTH)
            .map(|index| 1.1 + index as f32 / 40.0)
            .collect::<Vec<_>>();
        let epsilon = 1e-6;

        let mut hidden = Vec::new();
        embedding.row_dequant_into(1, &mut hidden);

        let original_hidden = hidden.clone();
        let mut residual = vec![f32::NAN; WIDTH];
        rms_norm_capture_mirror(&mut hidden, &mut residual, &first_gain, epsilon);
        assert_eq!(residual, original_hidden);
        assert_eq!(
            hidden,
            rms_norm_mirror(&original_hidden, &first_gain, epsilon, 1, WIDTH)
        );
        let first_contribution = first_dense.matvec(&hidden);
        residual_complete_mirror(&mut hidden, &residual, &first_contribution);
        assert_eq!(
            hidden,
            original_hidden
                .iter()
                .zip(&first_contribution)
                .map(|(&residual, &contribution)| residual + contribution)
                .collect::<Vec<_>>()
        );

        let before_second_norm = hidden.clone();
        rms_norm_capture_mirror(&mut hidden, &mut residual, &second_gain, epsilon);
        assert_eq!(residual, before_second_norm);
        let second_contribution = second_dense.matvec(&hidden);
        residual_complete_mirror(&mut hidden, &residual, &second_contribution);
        let before_final_norm = hidden.clone();
        let residual_before_final_norm = residual.clone();
        let expected_final_hidden =
            rms_norm_mirror(&before_final_norm, &final_gain, epsilon, 1, WIDTH);
        hidden = rms_norm_mirror(&hidden, &final_gain, epsilon, 1, WIDTH);

        assert_eq!(residual, residual_before_final_norm);
        assert_eq!(hidden, expected_final_hidden);
        assert_ne!(hidden, before_final_norm);
        assert_eq!(before_final_norm.len(), WIDTH);
        assert_eq!(hidden.len(), WIDTH);
        assert!(hidden.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn registry_missing_duplicate_kind_shape_and_context_checks_fail_closed() {
        let f32_layout =
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, 3, 5, 60).unwrap();
        let key = GpuNativeDenseWeightKey::try_new("embedding").unwrap();
        let mut registry = GpuNativeDenseWeightRegistry::new(41);
        assert_eq!(
            registry.handle_for(&key, GpuNativeDenseWeightKind::F32, 3, 5),
            Err(GpuNativeBootstrapError::MissingDenseWeight {
                key: "embedding".to_string(),
            })
        );

        let handle = registry
            .insert(test_weight(7, "embedding", f32_layout, ()))
            .unwrap();
        assert_eq!(handle.layout().kind(), GpuNativeDenseWeightKind::F32);
        assert_eq!(
            registry.insert(test_weight(8, "embedding", f32_layout, ())),
            Err(GpuNativeBootstrapError::DuplicateDenseWeight {
                key: "embedding".to_string(),
            })
        );
        assert_eq!(
            registry.handle_for(&key, GpuNativeDenseWeightKind::Q8_0, 3, 5),
            Err(GpuNativeBootstrapError::DenseWeightKindMismatch {
                key: "embedding".to_string(),
                expected: GpuNativeDenseWeightKind::Q8_0,
                actual: GpuNativeDenseWeightKind::F32,
            })
        );
        assert_eq!(
            registry.handle_for(&key, GpuNativeDenseWeightKind::F32, 4, 5),
            Err(GpuNativeBootstrapError::DenseWeightShapeMismatch {
                key: "embedding".to_string(),
                expected_rows: 4,
                expected_cols: 5,
                actual_rows: 3,
                actual_cols: 5,
            })
        );

        let other_registry = GpuNativeDenseWeightRegistry::<()>::new(42);
        assert!(matches!(
            other_registry.resolve(&handle),
            Err(GpuNativeBootstrapError::ForeignDenseWeightHandle)
        ));
        let mut stale = handle;
        stale.weight_id += 1;
        assert!(matches!(
            registry.resolve(&stale),
            Err(GpuNativeBootstrapError::StaleDenseWeightHandle { .. })
        ));
    }

    #[test]
    fn q8_shader_mirror_matches_repository_dequant_and_signed_scale_semantics() {
        let mut bytes = vec![0u8; 2 * Q8_0_BLOCK_BYTES];
        for (block, scale) in [-0.5f32, 0.25f32].into_iter().enumerate() {
            let offset = block * Q8_0_BLOCK_BYTES;
            bytes[offset..offset + 2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
            for i in 0..Q8_0_BLOCK_ELEMS {
                bytes[offset + 2 + i] = (i as i8 - 16) as u8;
            }
            let mut expected = [0.0; Q8_0_BLOCK_ELEMS];
            dequantize_q8_0_block(&bytes[offset..offset + Q8_0_BLOCK_BYTES], &mut expected);
            for (i, &expected) in expected.iter().enumerate() {
                assert_eq!(
                    read_q8_mirror(&bytes, block * Q8_0_BLOCK_ELEMS + i),
                    expected
                );
            }
        }
    }

    #[test]
    fn f32_and_q8_host_mirrors_match_dense_weight_cpu_matvec() {
        let rows = 3;
        let cols = 35;
        let values = (0..rows * cols)
            .map(|i| ((i * 17 % 43) as f32 - 21.0) / 7.0)
            .collect::<Vec<_>>();
        let input = (0..cols)
            .map(|i| ((i * 11 % 19) as f32 - 9.0) / 5.0)
            .collect::<Vec<_>>();

        let f32_weight = DenseWeight::from_f32(values.clone(), rows, cols);
        let mut f32_mirror = vec![0.0; rows];
        for row in 0..rows {
            for col in 0..cols {
                f32_mirror[row] += values[row * cols + col] * input[col];
            }
        }
        assert_close(&f32_mirror, &f32_weight.matvec(&input), 1e-5);

        let bytes = q8_bytes(&values);
        let q8_weight = DenseWeight::from_q8_0_bytes(bytes.clone(), rows, cols).unwrap();
        let q8_mirror = q8_gemv_mirror(&bytes, rows, cols, &input);
        assert_close(&q8_mirror, &q8_weight.matvec(&input), 1e-5);
    }

    #[test]
    fn embedding_bounds_and_row_mirror_cover_first_middle_last_and_invalid() {
        let rows = 5;
        let cols = 7;
        let values = (0..rows * cols)
            .map(|i| i as f32 - 10.0)
            .collect::<Vec<_>>();
        let bytes = q8_bytes(&values);
        let weight = DenseWeight::from_q8_0_bytes(bytes.clone(), rows, cols).unwrap();
        let layout = GpuNativeDenseWeightLayout::from_weight(&weight).unwrap();
        for token in [0u32, 2, 4] {
            layout.validate_embedding_token(token).unwrap();
            let mut expected = Vec::new();
            weight.row_dequant_into(token as usize, &mut expected);
            let actual = (0..cols)
                .map(|col| read_q8_mirror(&bytes, token as usize * cols + col))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
        assert_eq!(
            layout.validate_embedding_token(5),
            Err(GpuNativeBootstrapError::InvalidEmbeddingToken {
                token_id: 5,
                vocab_size: 5,
            })
        );
    }

    #[test]
    fn registration_and_dispatch_counters_keep_uploads_distinct() {
        let counters = GpuNativeExecutionCounters::default();
        counters.record_dense_weight_registration(3, 108);
        let after_registration = counters.snapshot();
        assert_eq!(after_registration.dense_weights_registered, 1);
        assert_eq!(after_registration.dense_weight_chunks, 3);
        assert_eq!(after_registration.dense_weight_uploads, 3);
        assert_eq!(after_registration.dense_weight_upload_bytes, 108);
        assert_eq!(after_registration.dense_weight_resident_bytes, 108);
        assert_eq!(after_registration.dense_gemv_dispatches, 0);
        assert_eq!(after_registration.dense_gemv_chunk_dispatches, 0);

        counters.record_dense_gemv_dispatch(3);
        counters.record_dense_gemv_dispatch(2);
        counters.record_embedding_dispatch();
        counters.record_rms_norm_state_dispatch(1);
        counters.record_rms_norm_state_dispatch(1);
        counters.record_rms_norm_scratch_dispatch(4);
        counters.record_residual_add_dispatch();
        counters.record_residual_add_dispatch();
        counters.record_rope_registration(8);
        counters.record_attention_prepare_dispatch();
        counters.record_rope_dispatch(6);
        counters.record_rope_dispatch(2);
        counters.record_kv_append();
        counters.record_causal_attention_dispatch();
        counters.record_attention_complete_dispatch();
        let after_dispatch = counters.snapshot();
        assert_eq!(after_dispatch.dense_weight_uploads, 3);
        assert_eq!(after_dispatch.dense_weight_upload_bytes, 108);
        assert_eq!(after_dispatch.dense_gemv_dispatches, 2);
        assert_eq!(after_dispatch.dense_gemv_chunk_dispatches, 5);
        assert_eq!(after_dispatch.embedding_dispatches, 1);
        assert_eq!(after_dispatch.rms_norm_dispatches, 3);
        assert_eq!(after_dispatch.rms_norm_groups, 6);
        assert_eq!(after_dispatch.rms_norm_state_dispatches, 2);
        assert_eq!(after_dispatch.rms_norm_scratch_dispatches, 1);
        assert_eq!(after_dispatch.residual_add_dispatches, 2);
        assert_eq!(after_dispatch.rope_parameters_registered, 1);
        assert_eq!(after_dispatch.rope_parameter_uploads, 1);
        assert_eq!(after_dispatch.rope_parameter_upload_bytes, 8);
        assert_eq!(after_dispatch.attention_prepare_dispatches, 1);
        assert_eq!(after_dispatch.q_projection_dispatches, 1);
        assert_eq!(after_dispatch.k_projection_dispatches, 1);
        assert_eq!(after_dispatch.v_projection_dispatches, 1);
        assert_eq!(after_dispatch.rope_dispatches, 2);
        assert_eq!(after_dispatch.rope_groups, 8);
        assert_eq!(after_dispatch.kv_appends, 1);
        assert_eq!(after_dispatch.causal_attention_dispatches, 1);
        assert_eq!(after_dispatch.o_projection_dispatches, 1);
        assert_eq!(after_dispatch.attention_complete_dispatches, 1);
    }

    #[test]
    fn gpu_native_shaders_parse_and_validate_without_hardware() {
        for (source, entry_points) in [
            (
                GPU_NATIVE_DENSE_GEMV_SHADER,
                &["f32_gemv_main", "q8_0_gemv_main"][..],
            ),
            (
                GPU_NATIVE_EMBEDDING_SHADER,
                &["f32_embedding_main", "q8_0_embedding_main"][..],
            ),
            (
                GPU_NATIVE_RMSNORM_SHADER,
                &[
                    "rms_norm_capture_main",
                    "rms_norm_in_place_main",
                    "residual_add_main",
                ][..],
            ),
            (GPU_NATIVE_ROPE_SHADER, &["rope_main"][..]),
            (GPU_NATIVE_KV_APPEND_SHADER, &["kv_append_main"][..]),
            (GPU_NATIVE_ATTENTION_SHADER, &["causal_attention_main"][..]),
            (GPU_NATIVE_TEST_COMPARE_SHADER, &["compare_main"][..]),
        ] {
            let module = naga::front::wgsl::parse_str(source).expect("GPU-native WGSL must parse");
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .expect("GPU-native WGSL must validate");
            for entry_point in entry_points {
                assert!(module
                    .entry_points
                    .iter()
                    .any(|entry| entry.name == *entry_point));
            }
        }
        assert!(!GPU_NATIVE_ATTENTION_SHADER.contains("layer_offset"));
        assert!(!GPU_NATIVE_ATTENTION_SHADER.contains("MAX_SEQ_LEN"));
        assert!(GPU_NATIVE_ATTENTION_SHADER.contains("KEY_CACHE"));
        assert!(GPU_NATIVE_ATTENTION_SHADER.contains("VALUE_CACHE"));
        assert!(GPU_NATIVE_ATTENTION_SHADER.contains("running_denominator"));
    }

    const GPU_NATIVE_TEST_COMPARE_SHADER: &str = r#"
struct PushConstants {
    elements: u32,
    tolerance_bits: u32,
    actual_offset: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> ACTUAL: array<f32>;
@group(0) @binding(1) var<storage, read> EXPECTED: array<f32>;
@group(0) @binding(2) var<storage, read_write> STATUS: array<atomic<u32>>;

@compute @workgroup_size(64, 1, 1)
fn compare_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= pc.elements) {
        return;
    }
    let difference = abs(ACTUAL[pc.actual_offset + gid.x] - EXPECTED[gid.x]);
    if (!(difference <= bitcast<f32>(pc.tolerance_bits))) {
        atomicStore(&STATUS[0], 1u);
    }
}
"#;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GpuNativeTestComparePushConstants {
        elements: u32,
        tolerance_bits: u32,
        actual_offset: u32,
    }

    fn create_test_compare_pipeline(
        device: &wgpu::Device,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_native_test_compare_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_native_test_compare_pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..12,
            }],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_test_compare_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_TEST_COMPARE_SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_native_test_compare_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "compare_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        (layout, pipeline)
    }

    fn create_test_expected_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        values: &[f32],
    ) -> wgpu::Buffer {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (values.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(values));
        buffer
    }

    fn encode_test_compare(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        layout: &wgpu::BindGroupLayout,
        pipeline: &wgpu::ComputePipeline,
        actual: &wgpu::Buffer,
        expected: &wgpu::Buffer,
        status: &wgpu::Buffer,
        elements: usize,
        tolerance: f32,
    ) {
        encode_test_compare_at(
            device, encoder, layout, pipeline, actual, expected, status, elements, tolerance, 0,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_test_compare_at(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        layout: &wgpu::BindGroupLayout,
        pipeline: &wgpu::ComputePipeline,
        actual: &wgpu::Buffer,
        expected: &wgpu::Buffer,
        status: &wgpu::Buffer,
        elements: usize,
        tolerance: f32,
        actual_offset: usize,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_test_compare_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: actual.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: expected.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: status.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_test_compare_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(
            0,
            bytemuck::bytes_of(&GpuNativeTestComparePushConstants {
                elements: elements as u32,
                tolerance_bits: tolerance.to_bits(),
                actual_offset: actual_offset as u32,
            }),
        );
        pass.dispatch_workgroups((elements as u32).div_ceil(GPU_NATIVE_WORKGROUP_SIZE), 1, 1);
    }

    /// Requires an actual hardware WGPU adapter. This intentionally uses the
    /// production execution-context resolver and maps only its four-byte
    /// aggregate validation status, never GPU-native hidden or scratch data.
    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_dense_gemv_embedding_persistence() {
        use super::super::{
            resolve_execution_context, ComputeOffload, GpuBackendGeometry, RoutedExpertGpuSpec,
        };
        use crate::inference::WeightDtype;

        const COLS: usize = 35;
        let expert_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(
            1024 * 1024,
            0.5,
            16,
        ));
        let execution = resolve_execution_context(
            ComputeOffload::Gpu,
            false,
            GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: 8,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 8,
                v_head_dim: 8,
                q4_truncation_tolerance: 0,
            },
            RoutedExpertGpuSpec {
                dtype: WeightDtype::F32,
                d_model: 32,
                d_ff: 64,
            },
            expert_cache,
        )
        .expect("L4 must construct the authoritative production GPU backend");
        let executor = execution
            .create_gpu_native_executor_context(COLS)
            .expect("GPU-native executor must retain the authoritative backend");
        let gpu = executor.authoritative_gpu().unwrap();

        let gemv_input_values = (0..COLS)
            .map(|i| ((i * 11 % 19) as f32 - 9.0) / 5.0)
            .collect::<Vec<_>>();
        let f32_gemv_values = (0..3 * COLS)
            .map(|i| ((i * 17 % 43) as f32 - 21.0) / 7.0)
            .collect::<Vec<_>>();
        let f32_gemv_weight = DenseWeight::from_f32(f32_gemv_values, 3, COLS);
        let q8_gemv_values = (0..3 * COLS)
            .map(|i| ((i * 23 % 47) as f32 - 23.0) / 9.0)
            .collect::<Vec<_>>();
        let q8_gemv_weight =
            DenseWeight::from_q8_0_bytes(q8_bytes(&q8_gemv_values), 3, COLS).unwrap();
        let f32_embedding_values = (0..5 * COLS)
            .map(|i| ((i * 13 % 41) as f32 - 20.0) / 6.0)
            .collect::<Vec<_>>();
        let f32_embedding_weight = DenseWeight::from_f32(f32_embedding_values, 5, COLS);
        let q8_embedding_values = (0..3 * COLS)
            .map(|i| ((i * 29 % 53) as f32 - 26.0) / 8.0)
            .collect::<Vec<_>>();
        let q8_embedding_weight =
            DenseWeight::from_q8_0_bytes(q8_bytes(&q8_embedding_values), 3, COLS).unwrap();

        let f32_gemv_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.f32_gemv").unwrap(),
                &f32_gemv_weight,
            )
            .unwrap();
        let q8_gemv_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.q8_gemv").unwrap(),
                &q8_gemv_weight,
            )
            .unwrap();
        let f32_embedding_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.f32_embedding").unwrap(),
                &f32_embedding_weight,
            )
            .unwrap();
        let q8_embedding_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.q8_embedding").unwrap(),
                &q8_embedding_weight,
            )
            .unwrap();
        let registered = executor.execution_snapshot();
        assert_eq!(registered.dense_weights_registered, 4);

        let input = executor.create_scratch(COLS).unwrap();
        let f32_output = executor.create_scratch(3).unwrap();
        let q8_output = executor.create_scratch(3).unwrap();
        let state = executor.create_token_state().unwrap();
        gpu.queue
            .write_buffer(&input.buffer, 0, bytemuck::cast_slice(&gemv_input_values));

        let f32_gemv_expected = create_test_expected_buffer(
            &gpu.device,
            &gpu.queue,
            "gpu_native_test_f32_gemv_expected",
            &f32_gemv_weight.matvec(&gemv_input_values),
        );
        let q8_gemv_expected = create_test_expected_buffer(
            &gpu.device,
            &gpu.queue,
            "gpu_native_test_q8_gemv_expected",
            &q8_gemv_weight.matvec(&gemv_input_values),
        );
        let f32_embedding_expected = [0usize, 2, 4]
            .into_iter()
            .map(|row| {
                let mut expected = Vec::new();
                f32_embedding_weight.row_dequant_into(row, &mut expected);
                create_test_expected_buffer(
                    &gpu.device,
                    &gpu.queue,
                    "gpu_native_test_f32_embedding_expected",
                    &expected,
                )
            })
            .collect::<Vec<_>>();
        let q8_embedding_expected = [0usize, 1, 2]
            .into_iter()
            .map(|row| {
                let mut expected = Vec::new();
                q8_embedding_weight.row_dequant_into(row, &mut expected);
                create_test_expected_buffer(
                    &gpu.device,
                    &gpu.queue,
                    "gpu_native_test_q8_embedding_expected",
                    &expected,
                )
            })
            .collect::<Vec<_>>();

        let status = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_test_validation_status"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&status, 0, &[0; 4]);
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_test_validation_staging"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (compare_layout, compare_pipeline) = create_test_compare_pipeline(&gpu.device);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_native_test_live_l4_encoder"),
            });

        for _ in 0..2 {
            executor
                .encode_dense_gemv_scratch_to_scratch(
                    &mut encoder,
                    &f32_gemv_handle,
                    &input,
                    &f32_output,
                )
                .unwrap();
            encode_test_compare(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &f32_output.buffer,
                &f32_gemv_expected,
                &status,
                3,
                1e-4,
            );
            executor
                .encode_dense_gemv_scratch_to_scratch(
                    &mut encoder,
                    &q8_gemv_handle,
                    &input,
                    &q8_output,
                )
                .unwrap();
            encode_test_compare(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &q8_output.buffer,
                &q8_gemv_expected,
                &status,
                3,
                1e-4,
            );
        }

        for (token, expected) in [0u32, 2, 4].into_iter().zip(&f32_embedding_expected) {
            executor
                .encode_embedding_lookup(&mut encoder, &f32_embedding_handle, token, &state)
                .unwrap();
            encode_test_compare(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &state.hidden,
                expected,
                &status,
                COLS,
                1e-6,
            );
        }
        for (token, expected) in [0u32, 1, 2].into_iter().zip(&q8_embedding_expected) {
            executor
                .encode_embedding_lookup(&mut encoder, &q8_embedding_handle, token, &state)
                .unwrap();
            encode_test_compare(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &state.hidden,
                expected,
                &status,
                COLS,
                1e-6,
            );
        }

        let encoded = executor.execution_snapshot();
        assert_eq!(
            encoded.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(
            encoded.dense_weight_upload_bytes,
            registered.dense_weight_upload_bytes
        );
        assert_eq!(encoded.intermediate_maps, 0);
        assert_eq!(encoded.intermediate_readbacks, 0);
        encoder.copy_buffer_to_buffer(&status, 0, &staging, 0, GPU_NATIVE_STATUS_BYTES);
        gpu.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("validation map callback must be drained")
            .expect("validation status must map");
        let mapped = slice.get_mapped_range();
        let status_value = u32::from_le_bytes(mapped[..4].try_into().unwrap());
        drop(mapped);
        staging.unmap();
        assert_eq!(status_value, 0, "on-device GPU-native comparison failed");

        let completed = executor.execution_snapshot();
        assert_eq!(
            completed.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(
            completed.dense_weight_upload_bytes,
            registered.dense_weight_upload_bytes
        );
        assert_eq!(completed.intermediate_maps, 0);
        assert_eq!(completed.intermediate_readbacks, 0);
    }

    /// Requires an actual NVIDIA L4 WGPU adapter. All production operations
    /// share one caller-owned encoder; validation maps only a four-byte status.
    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_rmsnorm_residual_chain() {
        use super::super::{
            resolve_execution_context, ComputeOffload, GpuBackendGeometry, RoutedExpertGpuSpec,
        };
        use crate::inference::WeightDtype;

        const WIDTH: usize = 65;
        const GROUPS: usize = 3;
        const GROUP_WIDTH: usize = 67;
        const EPSILON: f32 = 1e-6;

        let expert_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(
            1024 * 1024,
            0.5,
            16,
        ));
        let execution = resolve_execution_context(
            ComputeOffload::Gpu,
            false,
            GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: 8,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 8,
                v_head_dim: 8,
                q4_truncation_tolerance: 0,
            },
            RoutedExpertGpuSpec {
                dtype: WeightDtype::F32,
                d_model: 32,
                d_ff: 64,
            },
            expert_cache,
        )
        .expect("L4 must construct the authoritative production GPU backend");
        let executor = execution
            .create_gpu_native_executor_context(WIDTH)
            .expect("GPU-native executor must retain the authoritative backend");
        let gpu = executor.authoritative_gpu().unwrap();

        let first_gain = (0..WIDTH)
            .map(|index| 0.75 + (index * 7 % 17) as f32 / 40.0)
            .collect::<Vec<_>>();
        let second_gain = (0..WIDTH)
            .map(|index| 0.9 + (index * 11 % 19) as f32 / 50.0)
            .collect::<Vec<_>>();
        let final_gain = (0..WIDTH)
            .map(|index| 1.05 - (index * 5 % 13) as f32 / 60.0)
            .collect::<Vec<_>>();
        let grouped_gain = (0..GROUP_WIDTH)
            .map(|index| 0.8 + (index * 13 % 23) as f32 / 55.0)
            .collect::<Vec<_>>();

        let first_norm = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.norm.first").unwrap(),
                &first_gain,
            )
            .unwrap();
        let second_norm = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.norm.second").unwrap(),
                &second_gain,
            )
            .unwrap();
        let final_norm = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.norm.final").unwrap(),
                &final_gain,
            )
            .unwrap();
        let grouped_norm = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.norm.grouped").unwrap(),
                &grouped_gain,
            )
            .unwrap();
        let norm_registered = executor.execution_snapshot();
        assert_eq!(norm_registered.dense_weights_registered, 4);
        assert_eq!(norm_registered.dense_weight_uploads, 4);

        let embedding = DenseWeight::from_f32(
            (0..3 * WIDTH)
                .map(|index| ((index * 13 % 43) as f32 - 21.0) / 8.0)
                .collect(),
            3,
            WIDTH,
        );
        let first_dense = DenseWeight::from_f32(
            (0..WIDTH * WIDTH)
                .map(|index| ((index * 17 % 47) as f32 - 23.0) / 97.0)
                .collect(),
            WIDTH,
            WIDTH,
        );
        let second_dense_values = (0..WIDTH * WIDTH)
            .map(|index| ((index * 19 % 53) as f32 - 26.0) / 89.0)
            .collect::<Vec<_>>();
        let second_dense =
            DenseWeight::from_q8_0_bytes(q8_bytes(&second_dense_values), WIDTH, WIDTH).unwrap();
        let embedding_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.chain.embedding").unwrap(),
                &embedding,
            )
            .unwrap();
        let first_dense_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.chain.first_dense").unwrap(),
                &first_dense,
            )
            .unwrap();
        let second_dense_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.chain.second_dense").unwrap(),
                &second_dense,
            )
            .unwrap();
        let registered = executor.execution_snapshot();
        assert_eq!(registered.dense_weights_registered, 7);
        assert_eq!(registered.dense_weight_uploads, 7);

        let mut expected_hidden = Vec::new();
        embedding.row_dequant_into(1, &mut expected_hidden);
        let first_residual = expected_hidden.clone();
        expected_hidden =
            crate::transformer::RmsNorm::new(first_gain, EPSILON).forward(&expected_hidden);
        let first_contribution = first_dense.matvec(&expected_hidden);
        expected_hidden = residual_add_mirror(&first_residual, &first_contribution);
        let expected_residual = expected_hidden.clone();
        expected_hidden =
            crate::transformer::RmsNorm::new(second_gain, EPSILON).forward(&expected_hidden);
        let second_contribution = second_dense.matvec(&expected_hidden);
        expected_hidden = residual_add_mirror(&expected_residual, &second_contribution);
        expected_hidden =
            crate::transformer::RmsNorm::new(final_gain, EPSILON).forward(&expected_hidden);

        let grouped_input = (0..GROUPS * GROUP_WIDTH)
            .map(|index| ((index * 23 % 59) as f32 - 29.0) / 11.0)
            .collect::<Vec<_>>();
        let grouped_expected = grouped_input
            .chunks_exact(GROUP_WIDTH)
            .flat_map(|group| {
                crate::transformer::RmsNorm::new(grouped_gain.clone(), EPSILON).forward(group)
            })
            .collect::<Vec<_>>();

        let state = executor.create_token_state().unwrap();
        let contribution = executor.create_scratch(WIDTH).unwrap();
        let grouped = executor.create_scratch(GROUPS * GROUP_WIDTH).unwrap();
        gpu.queue
            .write_buffer(&grouped.buffer, 0, bytemuck::cast_slice(&grouped_input));
        let expected_hidden_buffer = create_test_expected_buffer(
            &gpu.device,
            &gpu.queue,
            "gpu_native_rms_chain_expected_hidden",
            &expected_hidden,
        );
        let expected_residual_buffer = create_test_expected_buffer(
            &gpu.device,
            &gpu.queue,
            "gpu_native_rms_chain_expected_residual",
            &expected_residual,
        );
        let grouped_expected_buffer = create_test_expected_buffer(
            &gpu.device,
            &gpu.queue,
            "gpu_native_grouped_rms_expected",
            &grouped_expected,
        );
        let status = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_rms_chain_validation_status"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&status, 0, &[0; 4]);
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_rms_chain_validation_staging"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (compare_layout, compare_pipeline) = create_test_compare_pipeline(&gpu.device);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_native_rms_chain_live_l4_encoder"),
            });

        executor
            .encode_embedding_lookup(&mut encoder, &embedding_handle, 1, &state)
            .unwrap();
        executor
            .encode_rms_norm_state_in_place(&mut encoder, &first_norm, EPSILON, &state)
            .unwrap();
        executor
            .encode_dense_gemv_hidden_to_scratch(
                &mut encoder,
                &first_dense_handle,
                &state,
                &contribution,
            )
            .unwrap();
        executor
            .encode_residual_add_scratch_to_hidden(&mut encoder, &state, &contribution)
            .unwrap();
        executor
            .encode_rms_norm_state_in_place(&mut encoder, &second_norm, EPSILON, &state)
            .unwrap();
        executor
            .encode_dense_gemv_hidden_to_scratch(
                &mut encoder,
                &second_dense_handle,
                &state,
                &contribution,
            )
            .unwrap();
        executor
            .encode_residual_add_scratch_to_hidden(&mut encoder, &state, &contribution)
            .unwrap();
        executor
            .encode_rms_norm_hidden_in_place(&mut encoder, &final_norm, EPSILON, &state)
            .unwrap();
        executor
            .encode_rms_norm_scratch_in_place(
                &mut encoder,
                &grouped_norm,
                EPSILON,
                &grouped,
                GROUPS,
                GROUP_WIDTH,
            )
            .unwrap();

        for (actual, expected, elements) in [
            (&state.hidden, &expected_hidden_buffer, WIDTH),
            (&state.residual, &expected_residual_buffer, WIDTH),
            (
                &grouped.buffer,
                &grouped_expected_buffer,
                GROUPS * GROUP_WIDTH,
            ),
        ] {
            encode_test_compare(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                actual,
                expected,
                &status,
                elements,
                2e-3,
            );
        }

        let encoded = executor.execution_snapshot();
        assert_eq!(
            encoded.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(
            encoded.dense_weight_upload_bytes,
            registered.dense_weight_upload_bytes
        );
        assert_eq!(encoded.embedding_dispatches, 1);
        assert_eq!(encoded.dense_gemv_dispatches, 2);
        assert_eq!(encoded.rms_norm_dispatches, 4);
        assert_eq!(encoded.rms_norm_groups, 6);
        assert_eq!(encoded.rms_norm_state_dispatches, 3);
        assert_eq!(encoded.rms_norm_scratch_dispatches, 1);
        assert_eq!(encoded.residual_add_dispatches, 2);
        assert_eq!(encoded.queue_submissions, 0);
        assert_eq!(encoded.intermediate_maps, 0);
        assert_eq!(encoded.intermediate_readbacks, 0);

        encoder.copy_buffer_to_buffer(&status, 0, &staging, 0, GPU_NATIVE_STATUS_BYTES);
        gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("validation map callback must be drained")
            .expect("validation status must map");
        let mapped = slice.get_mapped_range();
        let status_value = u32::from_le_bytes(mapped[..4].try_into().unwrap());
        drop(mapped);
        staging.unmap();
        assert_eq!(status_value, 0, "on-device RMS/residual comparison failed");

        let completed = executor.execution_snapshot();
        assert_eq!(
            completed.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(completed.intermediate_maps, 0);
        assert_eq!(completed.intermediate_readbacks, 0);
    }

    /// Requires an actual NVIDIA L4 WGPU adapter. The production Q/K/V/KV
    /// buffers remain non-readable; validation maps one aggregate status word.
    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_attention_prepare_kv() {
        use super::super::{
            resolve_execution_context, ComputeOffload, GpuBackendGeometry, RoutedExpertGpuSpec,
        };
        use crate::inference::WeightDtype;

        const D_MODEL: usize = 6;
        const NUM_HEADS: usize = 4;
        const NUM_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        const ROPE_DIM: usize = 4;
        const MAX_SEQ_LEN: usize = 4;
        const EPSILON: f32 = 1e-6;
        let geometry = GpuNativeAttentionGeometry::try_new(
            D_MODEL,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            ROPE_DIM,
        )
        .unwrap();
        let expert_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(
            1024 * 1024,
            0.5,
            16,
        ));
        let execution = resolve_execution_context(
            ComputeOffload::Gpu,
            false,
            GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: MAX_SEQ_LEN,
                num_heads: NUM_HEADS,
                num_kv_heads: NUM_KV_HEADS,
                head_dim: HEAD_DIM,
                v_head_dim: HEAD_DIM,
                q4_truncation_tolerance: 0,
            },
            RoutedExpertGpuSpec {
                dtype: WeightDtype::F32,
                d_model: D_MODEL,
                d_ff: 8,
            },
            expert_cache,
        )
        .expect("L4 must construct the authoritative production GPU backend");
        let executor = execution
            .create_gpu_native_executor_context(D_MODEL)
            .expect("GPU-native executor must retain the authoritative backend");
        let gpu = executor.authoritative_gpu().unwrap();

        let q_projection = DenseWeight::from_f32(
            (0..geometry.q_width * D_MODEL)
                .map(|index| ((index * 11 % 37) as f32 - 18.0) / 19.0)
                .collect(),
            geometry.q_width,
            D_MODEL,
        );
        let k_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * D_MODEL)
                .map(|index| ((index * 13 % 41) as f32 - 20.0) / 17.0)
                .collect(),
            geometry.kv_width,
            D_MODEL,
        );
        let v_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * D_MODEL)
                .map(|index| ((index * 17 % 43) as f32 - 21.0) / 23.0)
                .collect(),
            geometry.kv_width,
            D_MODEL,
        );
        let o_projection = DenseWeight::from_f32(
            (0..D_MODEL * geometry.q_width)
                .map(|index| ((index * 19 % 47) as f32 - 23.0) / 29.0)
                .collect(),
            D_MODEL,
            geometry.q_width,
        );
        let q_gain = vec![0.8, 1.1, 0.9, 1.2];
        let k_gain = vec![1.3, 0.7, 1.0, 0.85];
        let q_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.attention.q").unwrap(),
                &q_projection,
            )
            .unwrap();
        let k_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.attention.k").unwrap(),
                &k_projection,
            )
            .unwrap();
        let v_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.attention.v").unwrap(),
                &v_projection,
            )
            .unwrap();
        let o_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.attention.o").unwrap(),
                &o_projection,
            )
            .unwrap();
        let q_norm_handle = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.attention.q_norm").unwrap(),
                &q_gain,
            )
            .unwrap();
        let k_norm_handle = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.attention.k_norm").unwrap(),
                &k_gain,
            )
            .unwrap();
        let rope_handle = executor
            .register_standard_rope(
                GpuNativeDenseWeightKey::try_new("test.attention.rope").unwrap(),
                ROPE_DIM,
                10_000.0,
            )
            .unwrap();
        let plan = executor
            .create_attention_plan(
                0,
                geometry,
                q_handle,
                k_handle,
                v_handle,
                o_handle,
                Some(GpuNativeAttentionNorm::try_new(q_norm_handle, EPSILON).unwrap()),
                Some(GpuNativeAttentionNorm::try_new(k_norm_handle, EPSILON).unwrap()),
                rope_handle,
            )
            .unwrap();
        let scratch = executor.create_attention_scratch(geometry).unwrap();
        let kv = executor
            .create_kv_state(1, MAX_SEQ_LEN, geometry.kv_width)
            .unwrap();
        let states = [
            executor.create_token_state().unwrap(),
            executor.create_token_state().unwrap(),
        ];
        let inputs = [
            vec![0.5, -1.0, 1.5, -2.0, 2.5, -3.0],
            vec![-0.25, 0.75, -1.25, 1.75, -2.25, 2.75],
        ];
        let positions = [1usize, 3usize];
        for (state, input) in states.iter().zip(&inputs) {
            gpu.queue
                .write_buffer(&state.hidden, 0, bytemuck::cast_slice(input));
        }
        let inverse_frequencies = [1.0, 0.01];
        let expected = inputs
            .iter()
            .zip(positions)
            .map(|(input, position)| {
                attention_prepare_mirror(
                    input,
                    geometry,
                    &q_projection,
                    &k_projection,
                    &v_projection,
                    Some((&q_gain, EPSILON)),
                    Some((&k_gain, EPSILON)),
                    &inverse_frequencies,
                    1.0,
                    position,
                )
            })
            .collect::<Vec<_>>();
        let expected_buffers = expected
            .iter()
            .enumerate()
            .map(|(index, (q, k, v))| {
                (
                    create_test_expected_buffer(
                        &gpu.device,
                        &gpu.queue,
                        &format!("gpu_native_attention_q_expected_{index}"),
                        q,
                    ),
                    create_test_expected_buffer(
                        &gpu.device,
                        &gpu.queue,
                        &format!("gpu_native_attention_k_expected_{index}"),
                        k,
                    ),
                    create_test_expected_buffer(
                        &gpu.device,
                        &gpu.queue,
                        &format!("gpu_native_attention_v_expected_{index}"),
                        v,
                    ),
                )
            })
            .collect::<Vec<_>>();

        let registered = executor.execution_snapshot();
        assert_eq!(registered.dense_weights_registered, 7);
        assert_eq!(registered.rope_parameters_registered, 1);
        assert_eq!(registered.rope_parameter_uploads, 1);
        let status = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_attention_validation_status"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&status, 0, &[0; 4]);
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_attention_validation_staging"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (compare_layout, compare_pipeline) = create_test_compare_pipeline(&gpu.device);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_native_attention_prepare_live_l4_encoder"),
            });

        for index in 0..states.len() {
            executor
                .encode_attention_prepare(
                    &mut encoder,
                    &plan,
                    &states[index],
                    &scratch,
                    &kv,
                    positions[index],
                )
                .unwrap();
            let (expected_q, expected_k, expected_v) = &expected_buffers[index];
            for (actual, expected_buffer, elements) in [
                (&scratch.q.buffer, expected_q, geometry.q_width),
                (&scratch.k.buffer, expected_k, geometry.kv_width),
                (&scratch.v.buffer, expected_v, geometry.kv_width),
            ] {
                encode_test_compare(
                    &gpu.device,
                    &mut encoder,
                    &compare_layout,
                    &compare_pipeline,
                    actual,
                    expected_buffer,
                    &status,
                    elements,
                    2e-3,
                );
            }
        }
        for index in 0..positions.len() {
            let offset = kv.layout.element_offset(0, positions[index]).unwrap();
            let (_, expected_k, expected_v) = &expected_buffers[index];
            encode_test_compare_at(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &kv.layers[0].key,
                expected_k,
                &status,
                geometry.kv_width,
                2e-3,
                offset,
            );
            encode_test_compare_at(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &kv.layers[0].value,
                expected_v,
                &status,
                geometry.kv_width,
                2e-3,
                offset,
            );
        }

        let encoded = executor.execution_snapshot();
        assert_eq!(
            encoded.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(
            encoded.dense_weight_upload_bytes,
            registered.dense_weight_upload_bytes
        );
        assert_eq!(
            encoded.rope_parameter_uploads,
            registered.rope_parameter_uploads
        );
        assert_eq!(
            encoded.rope_parameter_upload_bytes,
            registered.rope_parameter_upload_bytes
        );
        assert_eq!(encoded.dense_gemv_dispatches, 6);
        assert_eq!(encoded.rms_norm_dispatches, 4);
        assert_eq!(encoded.rms_norm_groups, 12);
        assert_eq!(encoded.attention_prepare_dispatches, 2);
        assert_eq!(encoded.q_projection_dispatches, 2);
        assert_eq!(encoded.k_projection_dispatches, 2);
        assert_eq!(encoded.v_projection_dispatches, 2);
        assert_eq!(encoded.rope_dispatches, 4);
        assert_eq!(encoded.rope_groups, 12);
        assert_eq!(encoded.kv_appends, 2);
        assert_eq!(encoded.queue_submissions, 0);
        assert_eq!(encoded.intermediate_maps, 0);
        assert_eq!(encoded.intermediate_readbacks, 0);

        encoder.copy_buffer_to_buffer(&status, 0, &staging, 0, GPU_NATIVE_STATUS_BYTES);
        gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("validation map callback must be drained")
            .expect("validation status must map");
        let mapped = slice.get_mapped_range();
        let status_value = u32::from_le_bytes(mapped[..4].try_into().unwrap());
        drop(mapped);
        staging.unmap();
        assert_eq!(status_value, 0, "on-device attention preparation failed");

        let completed = executor.execution_snapshot();
        assert_eq!(
            completed.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(
            completed.rope_parameter_uploads,
            registered.rope_parameter_uploads
        );
        assert_eq!(completed.intermediate_maps, 0);
        assert_eq!(completed.intermediate_readbacks, 0);
    }

    /// Requires an actual NVIDIA L4 WGPU adapter. The production attention,
    /// KV, context, projected, and token-state buffers remain non-readable;
    /// validation maps one aggregate status word after one queue submission.
    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_causal_attention_residual() {
        use super::super::{
            resolve_execution_context, ComputeOffload, GpuBackendGeometry, RoutedExpertGpuSpec,
        };
        use crate::inference::WeightDtype;

        const D_MODEL: usize = 6;
        const NUM_HEADS: usize = 4;
        const NUM_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 2;
        const ROPE_DIM: usize = 2;
        const MAX_SEQ_LEN: usize = 4;
        const EPSILON: f32 = 1e-6;
        let geometry = GpuNativeAttentionGeometry::try_new(
            D_MODEL,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            ROPE_DIM,
        )
        .unwrap();
        assert_ne!(geometry.q_width, D_MODEL);

        let expert_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(
            1024 * 1024,
            0.5,
            16,
        ));
        let execution = resolve_execution_context(
            ComputeOffload::Gpu,
            false,
            GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: MAX_SEQ_LEN,
                num_heads: NUM_HEADS,
                num_kv_heads: NUM_KV_HEADS,
                head_dim: HEAD_DIM,
                v_head_dim: HEAD_DIM,
                q4_truncation_tolerance: 0,
            },
            RoutedExpertGpuSpec {
                dtype: WeightDtype::F32,
                d_model: D_MODEL,
                d_ff: 8,
            },
            expert_cache,
        )
        .expect("L4 must construct the authoritative production GPU backend");
        let executor = execution
            .create_gpu_native_executor_context(D_MODEL)
            .expect("GPU-native executor must retain the authoritative backend");
        let gpu = executor.authoritative_gpu().unwrap();

        let q_projection = DenseWeight::from_f32(
            (0..geometry.q_width * D_MODEL)
                .map(|index| ((index * 11 % 37) as f32 - 18.0) / 19.0)
                .collect(),
            geometry.q_width,
            D_MODEL,
        );
        let k_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * D_MODEL)
                .map(|index| ((index * 13 % 41) as f32 - 20.0) / 17.0)
                .collect(),
            geometry.kv_width,
            D_MODEL,
        );
        let v_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * D_MODEL)
                .map(|index| ((index * 17 % 43) as f32 - 21.0) / 23.0)
                .collect(),
            geometry.kv_width,
            D_MODEL,
        );
        let o_projection = DenseWeight::from_f32(
            (0..D_MODEL * geometry.q_width)
                .map(|index| ((index * 19 % 47) as f32 - 23.0) / 29.0)
                .collect(),
            D_MODEL,
            geometry.q_width,
        );
        let q_gain = [0.8, 1.2];
        let k_gain = [1.1, 0.7];
        let q_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.causal_attention.q").unwrap(),
                &q_projection,
            )
            .unwrap();
        let k_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.causal_attention.k").unwrap(),
                &k_projection,
            )
            .unwrap();
        let v_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.causal_attention.v").unwrap(),
                &v_projection,
            )
            .unwrap();
        let o_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.causal_attention.o").unwrap(),
                &o_projection,
            )
            .unwrap();
        let q_norm_handle = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.causal_attention.q_norm").unwrap(),
                &q_gain,
            )
            .unwrap();
        let k_norm_handle = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.causal_attention.k_norm").unwrap(),
                &k_gain,
            )
            .unwrap();
        let rope_handle = executor
            .register_standard_rope(
                GpuNativeDenseWeightKey::try_new("test.causal_attention.rope").unwrap(),
                ROPE_DIM,
                10_000.0,
            )
            .unwrap();
        let plan = executor
            .create_attention_plan(
                0,
                geometry,
                q_handle,
                k_handle,
                v_handle,
                o_handle,
                Some(GpuNativeAttentionNorm::try_new(q_norm_handle, EPSILON).unwrap()),
                Some(GpuNativeAttentionNorm::try_new(k_norm_handle, EPSILON).unwrap()),
                rope_handle,
            )
            .unwrap();
        let scratch = executor.create_attention_scratch(geometry).unwrap();
        let poison_k = executor.create_scratch(geometry.kv_width).unwrap();
        let poison_v = executor.create_scratch(geometry.kv_width).unwrap();
        let kv = executor
            .create_kv_state(1, MAX_SEQ_LEN, geometry.kv_width)
            .unwrap();
        let states = [
            executor.create_token_state().unwrap(),
            executor.create_token_state().unwrap(),
            executor.create_token_state().unwrap(),
        ];
        let prepared_inputs = [
            vec![0.5, -1.0, 1.5, -2.0, 2.5, -3.0],
            vec![-0.25, 0.75, -1.25, 1.75, -2.25, 2.75],
            vec![1.1, -0.4, 0.9, -1.6, 2.2, -2.8],
        ];
        let saved_residuals = [
            vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            vec![-0.6, 0.5, -0.4, 0.3, -0.2, 0.1],
            vec![0.7, -0.8, 0.9, -1.0, 1.1, -1.2],
        ];
        for ((state, prepared), residual) in
            states.iter().zip(&prepared_inputs).zip(&saved_residuals)
        {
            gpu.queue
                .write_buffer(&state.hidden, 0, bytemuck::cast_slice(prepared));
            gpu.queue
                .write_buffer(&state.residual, 0, bytemuck::cast_slice(residual));
        }
        let future_key_poison = [10_000.0, -9_000.0, 8_000.0, -7_000.0];
        let future_value_poison = [50_000.0, -40_000.0, 30_000.0, -20_000.0];
        gpu.queue.write_buffer(
            &poison_k.buffer,
            0,
            bytemuck::cast_slice(&future_key_poison),
        );
        gpu.queue.write_buffer(
            &poison_v.buffer,
            0,
            bytemuck::cast_slice(&future_value_poison),
        );

        let inverse_frequencies = [1.0];
        let mut expected_keys = vec![0.0; MAX_SEQ_LEN * geometry.kv_width];
        let mut expected_values = vec![0.0; MAX_SEQ_LEN * geometry.kv_width];
        expected_keys[3 * geometry.kv_width..].copy_from_slice(&future_key_poison);
        expected_values[3 * geometry.kv_width..].copy_from_slice(&future_value_poison);
        let mut expected = Vec::new();
        for position in 0..3 {
            let (q, k, v) = attention_prepare_mirror(
                &prepared_inputs[position],
                geometry,
                &q_projection,
                &k_projection,
                &v_projection,
                Some((&q_gain, EPSILON)),
                Some((&k_gain, EPSILON)),
                &inverse_frequencies,
                1.0,
                position,
            );
            let offset = position * geometry.kv_width;
            expected_keys[offset..offset + geometry.kv_width].copy_from_slice(&k);
            expected_values[offset..offset + geometry.kv_width].copy_from_slice(&v);
            let (context, projected, hidden) = attention_complete_mirror(
                &q,
                &expected_keys,
                &expected_values,
                geometry,
                position + 1,
                &o_projection,
                &saved_residuals[position],
            );
            expected.push((q, k, v, context, projected, hidden));
        }
        let expected_buffers = expected
            .iter()
            .enumerate()
            .map(|(index, (_, _, _, context, projected, hidden))| {
                (
                    create_test_expected_buffer(
                        &gpu.device,
                        &gpu.queue,
                        &format!("gpu_native_causal_context_expected_{index}"),
                        context,
                    ),
                    create_test_expected_buffer(
                        &gpu.device,
                        &gpu.queue,
                        &format!("gpu_native_causal_projected_expected_{index}"),
                        projected,
                    ),
                    create_test_expected_buffer(
                        &gpu.device,
                        &gpu.queue,
                        &format!("gpu_native_causal_hidden_expected_{index}"),
                        hidden,
                    ),
                    create_test_expected_buffer(
                        &gpu.device,
                        &gpu.queue,
                        &format!("gpu_native_causal_residual_expected_{index}"),
                        &saved_residuals[index],
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected_key_buffers = (0..MAX_SEQ_LEN)
            .map(|position| {
                let offset = position * geometry.kv_width;
                create_test_expected_buffer(
                    &gpu.device,
                    &gpu.queue,
                    &format!("gpu_native_causal_key_expected_{position}"),
                    &expected_keys[offset..offset + geometry.kv_width],
                )
            })
            .collect::<Vec<_>>();
        let expected_value_buffers = (0..MAX_SEQ_LEN)
            .map(|position| {
                let offset = position * geometry.kv_width;
                create_test_expected_buffer(
                    &gpu.device,
                    &gpu.queue,
                    &format!("gpu_native_causal_value_expected_{position}"),
                    &expected_values[offset..offset + geometry.kv_width],
                )
            })
            .collect::<Vec<_>>();

        let registered = executor.execution_snapshot();
        assert_eq!(registered.dense_weights_registered, 7);
        assert_eq!(registered.rope_parameters_registered, 1);
        let status = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_causal_attention_validation_status"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&status, 0, &[0; 4]);
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_causal_attention_validation_staging"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (compare_layout, compare_pipeline) = create_test_compare_pipeline(&gpu.device);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_native_causal_attention_live_l4_encoder"),
            });

        executor
            .encode_kv_append(&mut encoder, &poison_k, &poison_v, &kv, 0, 3)
            .unwrap();
        for position in 0..3 {
            executor
                .encode_attention_prepare(
                    &mut encoder,
                    &plan,
                    &states[position],
                    &scratch,
                    &kv,
                    position,
                )
                .unwrap();
            executor
                .encode_attention_complete(
                    &mut encoder,
                    &plan,
                    &states[position],
                    &scratch,
                    &kv,
                    position,
                )
                .unwrap();
            let (context, projected, hidden, residual) = &expected_buffers[position];
            for (actual, expected_buffer, elements) in [
                (&scratch.context.buffer, context, geometry.q_width),
                (&scratch.projected.buffer, projected, geometry.d_model),
                (&states[position].hidden, hidden, geometry.d_model),
                (&states[position].residual, residual, geometry.d_model),
            ] {
                encode_test_compare(
                    &gpu.device,
                    &mut encoder,
                    &compare_layout,
                    &compare_pipeline,
                    actual,
                    expected_buffer,
                    &status,
                    elements,
                    3e-3,
                );
            }
        }
        for position in 0..MAX_SEQ_LEN {
            let offset = position * geometry.kv_width;
            encode_test_compare_at(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &kv.layers[0].key,
                &expected_key_buffers[position],
                &status,
                geometry.kv_width,
                3e-3,
                offset,
            );
            encode_test_compare_at(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &kv.layers[0].value,
                &expected_value_buffers[position],
                &status,
                geometry.kv_width,
                3e-3,
                offset,
            );
        }

        let encoded = executor.execution_snapshot();
        assert_eq!(
            encoded.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(
            encoded.dense_weight_upload_bytes,
            registered.dense_weight_upload_bytes
        );
        assert_eq!(
            encoded.rope_parameter_uploads,
            registered.rope_parameter_uploads
        );
        assert_eq!(encoded.dense_gemv_dispatches, 12);
        assert_eq!(encoded.rms_norm_dispatches, 6);
        assert_eq!(encoded.rms_norm_groups, 18);
        assert_eq!(encoded.attention_prepare_dispatches, 3);
        assert_eq!(encoded.q_projection_dispatches, 3);
        assert_eq!(encoded.k_projection_dispatches, 3);
        assert_eq!(encoded.v_projection_dispatches, 3);
        assert_eq!(encoded.rope_dispatches, 6);
        assert_eq!(encoded.rope_groups, 18);
        assert_eq!(encoded.kv_appends, 4);
        assert_eq!(encoded.causal_attention_dispatches, 3);
        assert_eq!(encoded.o_projection_dispatches, 3);
        assert_eq!(encoded.attention_complete_dispatches, 3);
        assert_eq!(encoded.residual_add_dispatches, 3);
        assert_eq!(encoded.queue_submissions, 0);
        assert_eq!(encoded.intermediate_maps, 0);
        assert_eq!(encoded.intermediate_readbacks, 0);
        assert_eq!(encoded.cpu_attention_calls, 0);
        assert_eq!(encoded.cpu_kv_mutations, 0);
        assert_eq!(encoded.cpu_layer_reentries, 0);

        encoder.copy_buffer_to_buffer(&status, 0, &staging, 0, GPU_NATIVE_STATUS_BYTES);
        gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("validation map callback must be drained")
            .expect("validation status must map");
        let mapped = slice.get_mapped_range();
        let status_value = u32::from_le_bytes(mapped[..4].try_into().unwrap());
        drop(mapped);
        staging.unmap();
        assert_eq!(status_value, 0, "on-device causal attention failed");

        let completed = executor.execution_snapshot();
        assert_eq!(
            completed.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(
            completed.rope_parameter_uploads,
            registered.rope_parameter_uploads
        );
        assert_eq!(completed.intermediate_maps, 0);
        assert_eq!(completed.intermediate_readbacks, 0);
        assert_eq!(completed.cpu_attention_calls, 0);
        assert_eq!(completed.cpu_kv_mutations, 0);
        assert_eq!(completed.cpu_layer_reentries, 0);
    }

    #[test]
    fn checked_layout_derives_qwen_vector_bytes() {
        let layout = GpuNativeTokenStateLayout::try_new(2048).expect("Qwen layout");
        assert_eq!(layout.d_model(), 2048);
        assert_eq!(layout.vector_bytes(), 8192);
        assert_eq!(layout.status_bytes(), 4);
        assert_eq!(layout.total_buffer_bytes(), 16_388);
    }

    #[test]
    fn checked_layout_rejects_zero_and_overflow() {
        assert_eq!(
            GpuNativeTokenStateLayout::try_new(0),
            Err(GpuNativeBootstrapError::InvalidDModel)
        );
        let overflowing_d_model = usize::MAX / std::mem::size_of::<f32>() + 1;
        assert_eq!(
            GpuNativeTokenStateLayout::try_new(overflowing_d_model),
            Err(GpuNativeBootstrapError::StateSizeOverflow {
                d_model: overflowing_d_model,
            })
        );
        let total_overflowing_d_model = usize::MAX / std::mem::size_of::<f32>();
        assert_eq!(
            GpuNativeTokenStateLayout::try_new(total_overflowing_d_model),
            Err(GpuNativeBootstrapError::StateSizeOverflow {
                d_model: total_overflowing_d_model,
            })
        );
    }

    #[test]
    fn layout_rejects_device_limit_incompatibility_before_allocation() {
        let layout = GpuNativeTokenStateLayout::try_new(2048).unwrap();
        let limits = wgpu::Limits {
            max_buffer_size: 4096,
            max_storage_buffer_binding_size: 4096,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            layout.validate_for_limits(&limits),
            Err(GpuNativeBootstrapError::Allocation(
                GpuStartupAllocationError::ExceedsMaxBufferSize {
                    label: "gpu_native_hidden".to_string(),
                    requested: 8192,
                    maximum: 4096,
                }
            ))
        );
    }

    #[test]
    fn intermediate_tensor_buffers_are_not_cpu_mappable() {
        let tensor = GpuNativeTokenStateLayout::tensor_usage();
        assert!(tensor.contains(wgpu::BufferUsages::STORAGE));
        assert!(tensor.contains(wgpu::BufferUsages::COPY_DST));
        assert!(!tensor.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!tensor.intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));

        let status = GpuNativeTokenStateLayout::status_usage();
        assert!(status.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!status.intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));

        for usage in [
            GpuNativeScratchLayout::usage(),
            GpuNativeDenseWeightLayout::usage(),
            GpuNativeKvLayout::usage(),
        ] {
            assert!(usage.contains(wgpu::BufferUsages::STORAGE));
            assert!(!usage.contains(wgpu::BufferUsages::COPY_SRC));
            assert!(!usage.intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));
        }
    }

    #[test]
    fn scratch_layout_is_checked_and_variable_width() {
        let q = GpuNativeScratchLayout::try_new(4096).unwrap();
        let kv = GpuNativeScratchLayout::try_new(512).unwrap();
        let router = GpuNativeScratchLayout::try_new(128).unwrap();
        assert_eq!((q.elements(), q.bytes()), (4096, 16_384));
        assert_eq!((kv.elements(), kv.bytes()), (512, 2048));
        assert_eq!((router.elements(), router.bytes()), (128, 512));
        assert_eq!(
            GpuNativeScratchLayout::try_new(0),
            Err(GpuNativeBootstrapError::InvalidScratchElements)
        );
    }

    #[derive(Debug)]
    struct DropProbe {
        allocation_id: u64,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn test_state(
        layout: GpuNativeTokenStateLayout,
        allocation_base: u64,
        drops: Arc<AtomicUsize>,
    ) -> GpuNativeTokenState<DropProbe> {
        GpuNativeTokenState::from_buffers(
            1,
            next_gpu_native_token_state_id(),
            layout,
            DropProbe {
                allocation_id: allocation_base,
                drops: drops.clone(),
            },
            DropProbe {
                allocation_id: allocation_base + 1,
                drops: drops.clone(),
            },
            DropProbe {
                allocation_id: allocation_base + 2,
                drops,
            },
        )
    }

    #[test]
    fn token_states_own_distinct_mutable_buffers_and_state_ids() {
        let layout = GpuNativeTokenStateLayout::try_new(32).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let first = test_state(layout, 10, drops.clone());
        let second = test_state(layout, 20, drops.clone());

        assert_ne!(first.state_id(), second.state_id());
        assert_eq!(first.layout(), second.layout());
        assert_ne!(first.hidden.allocation_id, second.hidden.allocation_id);
        assert_ne!(first.residual.allocation_id, second.residual.allocation_id);
        assert_ne!(first.status.allocation_id, second.status.allocation_id);
        drop(first);
        drop(second);
        assert_eq!(drops.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn token_state_cleanup_drops_each_owned_buffer_exactly_once() {
        let layout = GpuNativeTokenStateLayout::try_new(32).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let state = test_state(layout, 1, drops.clone());

        assert_eq!(drops.load(Ordering::Relaxed), 0);
        drop(state);
        assert_eq!(drops.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn request_local_kv_drops_each_per_layer_buffer_exactly_once() {
        let layout = GpuNativeKvLayout::try_new(2, 4, 8, &wgpu::Limits::default()).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let probe = |allocation_id| DropProbe {
            allocation_id,
            drops: drops.clone(),
        };
        let kv = GpuNativeKvState::from_layers(
            7,
            1,
            layout,
            vec![
                GpuNativeKvLayer {
                    key: probe(10),
                    value: probe(11),
                },
                GpuNativeKvLayer {
                    key: probe(20),
                    value: probe(21),
                },
            ],
        );
        assert_eq!(kv.layout(), layout);
        assert_ne!(
            kv.layers[0].key.allocation_id,
            kv.layers[0].value.allocation_id
        );
        assert_ne!(
            kv.layers[0].key.allocation_id,
            kv.layers[1].key.allocation_id
        );
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        drop(kv);
        assert_eq!(drops.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn request_state_ownership_is_separate_from_model_weight_registry() {
        let layout = GpuNativeTokenStateLayout::try_new(32).unwrap();
        let state_drops = Arc::new(AtomicUsize::new(0));
        let weight_drops = Arc::new(AtomicUsize::new(0));
        let state = test_state(layout, 1, state_drops.clone());
        let weight_layout =
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, 2, 2, 16).unwrap();
        let mut registry = GpuNativeDenseWeightRegistry::new(1);
        registry
            .insert(test_weight(
                1,
                "model.weight",
                weight_layout,
                DropProbe {
                    allocation_id: 100,
                    drops: weight_drops.clone(),
                },
            ))
            .unwrap();

        drop(state);
        assert_eq!(state_drops.load(Ordering::Relaxed), 3);
        assert_eq!(weight_drops.load(Ordering::Relaxed), 0);
        drop(registry);
        assert_eq!(weight_drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn initial_execution_snapshot_is_all_zero() {
        let counters = GpuNativeExecutionCounters::default();
        assert_eq!(counters.snapshot(), GpuNativeExecutionSnapshot::default());
    }

    #[test]
    fn cpu_execution_context_cannot_construct_gpu_native_executor() {
        let context = super::super::cpu_execution_context();
        assert!(matches!(
            context.create_gpu_native_executor_context(2048),
            Err(GpuNativeBootstrapError::GpuBackendUnavailable)
        ));
    }

    #[test]
    fn existing_operator_and_backend_modes_remain_distinct_from_bootstrap() {
        use super::super::{ComputeOffload, GpuBackendMode};

        let operator_mode_name = |mode| match mode {
            ComputeOffload::Cpu => "cpu",
            ComputeOffload::Gpu => "gpu",
            ComputeOffload::Auto => "auto",
            ComputeOffload::Hybrid => "hybrid",
        };
        assert_eq!(
            [
                ComputeOffload::Cpu,
                ComputeOffload::Gpu,
                ComputeOffload::Auto,
                ComputeOffload::Hybrid,
            ]
            .map(operator_mode_name),
            ["cpu", "gpu", "auto", "hybrid"]
        );

        let resource_mode_name = |mode| match mode {
            GpuBackendMode::RoutedExpertsOnly => "routed-experts-only",
            GpuBackendMode::Full => "legacy-full-resources",
        };
        assert_eq!(
            resource_mode_name(GpuBackendMode::Full),
            "legacy-full-resources"
        );
    }

    #[test]
    fn hardware_independent_test_backend_cannot_fake_gpu_native_executor() {
        let backend = Arc::new(BackendBox::TestGpu(super::super::TestGpuBackend::success(
            1.0,
        )));
        assert!(matches!(
            GpuNativeExecutorContext::try_new(backend, 2048),
            Err(GpuNativeBootstrapError::GpuBackendUnavailable)
        ));
    }

    #[test]
    fn snapshot_distinguishes_token_boundary_and_intermediate_readback() {
        let counters = GpuNativeExecutionCounters::default();
        counters.record_token_boundary_readback();
        let boundary = counters.snapshot();
        assert_eq!(boundary.token_boundary_readbacks, 1);
        assert_eq!(boundary.intermediate_readbacks, 0);

        counters.record_intermediate_readback();
        let intermediate = counters.snapshot();
        assert_eq!(intermediate.token_boundary_readbacks, 1);
        assert_eq!(intermediate.intermediate_readbacks, 1);
    }
}
