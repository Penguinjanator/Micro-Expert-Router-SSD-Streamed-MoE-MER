//! Isolated foundations for a future GPU-native token loop.
//!
//! This module is not reachable from current operator-facing execution modes.
//! It owns request-local device state, persistent dense model weights, and
//! encoder-only embedding/GEMV primitives on the authoritative WGPU device.
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
const GPU_NATIVE_WORKGROUP_SIZE: u32 = 64;

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
    DispatchGeometryUnsupported {
        workgroups: u64,
        maximum: u32,
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
            Self::DispatchGeometryUnsupported {
                workgroups,
                maximum,
            } => write!(
                f,
                "GPU-native dispatch requires {workgroups} workgroups, exceeding device maximum {maximum}"
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

struct GpuNativeDenseWeight<B = wgpu::Buffer> {
    weight_id: u64,
    key: GpuNativeDenseWeightKey,
    layout: GpuNativeDenseWeightLayout,
    buffer: B,
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

/// Immutable, serializable evidence for GPU-native execution boundaries.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct GpuNativeExecutionSnapshot {
    pub(crate) dense_weights_registered: u64,
    pub(crate) dense_weight_uploads: u64,
    pub(crate) dense_weight_upload_bytes: u64,
    pub(crate) dense_weight_resident_bytes: u64,
    pub(crate) dense_gemv_dispatches: u64,
    pub(crate) embedding_dispatches: u64,
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
    dense_weight_uploads: AtomicU64,
    dense_weight_upload_bytes: AtomicU64,
    dense_weight_resident_bytes: AtomicU64,
    dense_gemv_dispatches: AtomicU64,
    embedding_dispatches: AtomicU64,
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
            dense_weight_uploads: self.dense_weight_uploads.load(Ordering::Relaxed),
            dense_weight_upload_bytes: self.dense_weight_upload_bytes.load(Ordering::Relaxed),
            dense_weight_resident_bytes: self.dense_weight_resident_bytes.load(Ordering::Relaxed),
            dense_gemv_dispatches: self.dense_gemv_dispatches.load(Ordering::Relaxed),
            embedding_dispatches: self.embedding_dispatches.load(Ordering::Relaxed),
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

    fn record_dense_weight_registration(&self, allocation_bytes: u64) {
        self.dense_weights_registered
            .fetch_add(1, Ordering::Relaxed);
        self.dense_weight_uploads.fetch_add(1, Ordering::Relaxed);
        self.dense_weight_upload_bytes
            .fetch_add(allocation_bytes, Ordering::Relaxed);
        self.dense_weight_resident_bytes
            .fetch_add(allocation_bytes, Ordering::Relaxed);
    }

    fn record_dense_gemv_dispatch(&self) {
        self.dense_gemv_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    fn record_embedding_dispatch(&self) {
        self.embedding_dispatches.fetch_add(1, Ordering::Relaxed);
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
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeEmbeddingPushConstants {
    token_id: u32,
    rows: u32,
    cols: u32,
    _pad: u32,
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

        Ok(Self {
            context_id,
            authoritative_backend,
            device_identity,
            layout,
            dense_weights: ParkingMutex::new(GpuNativeDenseWeightRegistry::new(context_id)),
            dense_pipelines,
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
    /// exactly once. This is the only dense-weight upload path in the
    /// GPU-native plane; encoded GEMV and embedding calls only bind this
    /// persistent buffer.
    pub(crate) fn register_dense_weight(
        &self,
        key: GpuNativeDenseWeightKey,
        weight: &DenseWeight,
    ) -> Result<GpuNativeDenseWeightHandle, GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        let layout = GpuNativeDenseWeightLayout::from_weight(weight)?;
        let label = format!("gpu_native_dense_weight_{}", key.as_str());
        layout.validate_for_limits(&label, &gpu.device.limits())?;

        // Serialize the duplicate check through insertion so two startup
        // registrars cannot both upload the same stable key.
        let mut registry = self.dense_weights.lock();
        if registry.weights.contains_key(&key) {
            return Err(GpuNativeBootstrapError::DuplicateDenseWeight {
                key: key.as_str().to_string(),
            });
        }
        let buffer = create_startup_buffer(
            &gpu.device,
            &label,
            layout.allocation_bytes,
            GpuNativeDenseWeightLayout::usage(),
        )?;
        match weight {
            DenseWeight::F32 { values, .. } => {
                gpu.queue
                    .write_buffer(&buffer, 0, bytemuck::cast_slice(values));
            }
            DenseWeight::Q8_0 { bytes, .. } if bytes.len() as u64 == layout.allocation_bytes => {
                gpu.queue.write_buffer(&buffer, 0, bytes);
            }
            DenseWeight::Q8_0 { bytes, .. } => {
                let mut upload = Vec::with_capacity(layout.allocation_bytes as usize);
                upload.extend_from_slice(bytes);
                upload.resize(layout.allocation_bytes as usize, 0);
                gpu.queue.write_buffer(&buffer, 0, &upload);
            }
        }
        let registered = GpuNativeDenseWeight {
            weight_id: next_nonzero_id(&NEXT_GPU_NATIVE_WEIGHT_ID, "dense weight"),
            key,
            layout,
            buffer,
        };
        let handle = registry.insert(registered)?;
        self.counters
            .record_dense_weight_registration(layout.allocation_bytes);
        Ok(handle)
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
        let workgroups = self.checked_workgroups(weight.layout.rows, &gpu.device.limits())?;
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_dense_gemv_bind_group"),
            layout: &self.dense_pipelines.gemv_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weight.buffer.as_entire_binding(),
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
        let pipeline = match weight.layout.kind {
            GpuNativeDenseWeightKind::F32 => &self.dense_pipelines.f32_gemv,
            GpuNativeDenseWeightKind::Q8_0 => &self.dense_pipelines.q8_0_gemv,
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_dense_gemv_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(
            0,
            bytemuck::bytes_of(&GpuNativeGemvPushConstants {
                rows: weight.layout.rows as u32,
                cols: weight.layout.cols as u32,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
        self.counters.record_dense_gemv_dispatch();
        Ok(())
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
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_embedding_bind_group"),
            layout: &self.dense_pipelines.embedding_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weight.buffer.as_entire_binding(),
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
                token_id,
                rows: weight.layout.rows as u32,
                cols: weight.layout.cols as u32,
                _pad: 0,
            }),
        );
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
        self.counters.record_embedding_dispatch();
        Ok(())
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

    fn test_weight<B>(
        weight_id: u64,
        key: &str,
        layout: GpuNativeDenseWeightLayout,
        buffer: B,
    ) -> GpuNativeDenseWeight<B> {
        GpuNativeDenseWeight {
            weight_id,
            key: GpuNativeDenseWeightKey::try_new(key).unwrap(),
            layout,
            buffer,
        }
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
        counters.record_dense_weight_registration(36);
        let after_registration = counters.snapshot();
        assert_eq!(after_registration.dense_weights_registered, 1);
        assert_eq!(after_registration.dense_weight_uploads, 1);
        assert_eq!(after_registration.dense_weight_upload_bytes, 36);
        assert_eq!(after_registration.dense_weight_resident_bytes, 36);
        assert_eq!(after_registration.dense_gemv_dispatches, 0);

        counters.record_dense_gemv_dispatch();
        counters.record_dense_gemv_dispatch();
        counters.record_embedding_dispatch();
        let after_dispatch = counters.snapshot();
        assert_eq!(after_dispatch.dense_weight_uploads, 1);
        assert_eq!(after_dispatch.dense_weight_upload_bytes, 36);
        assert_eq!(after_dispatch.dense_gemv_dispatches, 2);
        assert_eq!(after_dispatch.embedding_dispatches, 1);
    }

    #[test]
    fn gpu_native_dense_shaders_parse_and_validate_without_hardware() {
        for (source, entry_points) in [
            (
                GPU_NATIVE_DENSE_GEMV_SHADER,
                &["f32_gemv_main", "q8_0_gemv_main"][..],
            ),
            (
                GPU_NATIVE_EMBEDDING_SHADER,
                &["f32_embedding_main", "q8_0_embedding_main"][..],
            ),
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
