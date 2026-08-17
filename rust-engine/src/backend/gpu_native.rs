//! Bootstrap ownership for a future GPU-native token loop.
//!
//! This module deliberately contains no transformer arithmetic and is not
//! reachable from the current operator-facing execution modes. It establishes
//! only request-local device state, checked layout arithmetic, authoritative
//! device reuse, and execution evidence that later slices can build on.

// Slice 1 is intentionally not reachable from production token entrypoints.
#![allow(dead_code)]

use super::{create_startup_buffer, BackendBox, GpuDeviceIdentity, GpuStartupAllocationError};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const GPU_NATIVE_STATUS_BYTES: u64 = std::mem::size_of::<u32>() as u64;

/// Typed, fail-closed construction failure for the GPU-native bootstrap.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeBootstrapError {
    GpuBackendUnavailable,
    DeviceLost { detail: String },
    InvalidDModel,
    StateSizeOverflow { d_model: usize },
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
    state_id: u64,
    layout: GpuNativeTokenStateLayout,
    hidden: B,
    residual: B,
    status: B,
}

impl<B> GpuNativeTokenState<B> {
    fn from_buffers(
        state_id: u64,
        layout: GpuNativeTokenStateLayout,
        hidden: B,
        residual: B,
        status: B,
    ) -> Self {
        Self {
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

/// Internal bootstrap for future GPU-owned token execution.
///
/// Retaining the exact `Arc<BackendBox>` keeps the authoritative non-cloneable
/// WGPU `Device` and `Queue` alive. It does not request or select hardware and
/// it is intentionally absent from all current execution-plan resolution.
pub(crate) struct GpuNativeExecutorContext {
    authoritative_backend: Arc<BackendBox>,
    device_identity: GpuDeviceIdentity,
    layout: GpuNativeTokenStateLayout,
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

        Ok(Self {
            authoritative_backend,
            device_identity,
            layout,
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
            state_id,
            self.layout,
            hidden,
            residual,
            status,
        ))
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
    use std::sync::atomic::AtomicUsize;

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
