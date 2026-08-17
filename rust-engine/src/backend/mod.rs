//! Math-backend module connecting GPU execution via wgpu and providing a CPU fallback.

use anyhow::{anyhow, Result};
use parking_lot::Mutex as ParkingMutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Maximum routed-expert workspace dimension supported for both `d_model` and
/// `d_ff`. Sized for Mixtral-8x7B (`d_ff = 14336`).
const MAX_EXPERT_D_FF: usize = 16_384;
const DENSE_WORK_MAX_ELEMS: usize = 4096 * 4096;
const DENSE_BUFFER_BYTES: u64 = (DENSE_WORK_MAX_ELEMS * std::mem::size_of::<f32>()) as u64;

// Embed WGSL shaders using include_str
const MATMUL_SHADER: &str = include_str!("wgpu_shaders/matmul.wgsl");
const MATMUL_Q4_0_SHADER: &str = include_str!("wgpu_shaders/matmul_q4_0.wgsl");
const SWIGLU_SHADER: &str = include_str!("wgpu_shaders/swiglu.wgsl");
const SOFTMAX_SHADER: &str = include_str!("wgpu_shaders/softmax.wgsl");
const ATTENTION_SHADER: &str = include_str!("wgpu_shaders/attention.wgsl");

/// Explicit opt-in for treating software adapters (llvmpipe, SwiftShader,
/// WARP, etc.) as an allowed wgpu plane. The normal `--gpu` path should
/// never report these as a real GPU benchmark.
const ALLOW_SOFTWARE_WGPU_ADAPTER_ENV: &str = "MER_WGPU_ALLOW_SOFTWARE_ADAPTER";

/// Explicit construction mode for the authoritative wgpu backend.
///
/// `RoutedExpertsOnly` is the strict Hybrid capability set: it owns the
/// adapter/device and routed-expert resources, but no dense, attention, or KV
/// resources. `Full` preserves the existing legacy GPU resource set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuBackendMode {
    RoutedExpertsOnly,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCapability {
    RoutedExperts,
    Dense,
    Attention,
    Kv,
}

/// Typed fail-closed error returned when code calls a GPU component that the
/// authoritative execution plan deliberately left on CPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuCapabilityUnavailable {
    pub mode: GpuBackendMode,
    pub capability: GpuCapability,
}

impl fmt::Display for GpuCapabilityUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPU capability {:?} is unavailable in {:?} backend mode",
            self.capability, self.mode
        )
    }
}

impl std::error::Error for GpuCapabilityUnavailable {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GpuResourceInvariantError {
    mode: GpuBackendMode,
    resource: &'static str,
}

impl fmt::Display for GpuResourceInvariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPU resource plan for {:?} requires {}, but it was not constructed",
            self.mode, self.resource
        )
    }
}

impl std::error::Error for GpuResourceInvariantError {}

/// Typed startup allocation error emitted before `Device::create_buffer`, so
/// oversized planned resources cannot reach wgpu's uncaptured-error handler.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuStartupAllocationError {
    ExceedsMaxBufferSize {
        label: String,
        requested: u64,
        maximum: u64,
    },
    ExceedsMaxStorageBindingSize {
        label: String,
        requested: u64,
        maximum: u64,
    },
}

impl fmt::Display for GpuStartupAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExceedsMaxBufferSize {
                label,
                requested,
                maximum,
            } => write!(
                f,
                "GPU startup buffer {label:?} requests {requested} bytes, exceeding device max_buffer_size {maximum}"
            ),
            Self::ExceedsMaxStorageBindingSize {
                label,
                requested,
                maximum,
            } => write!(
                f,
                "GPU startup storage buffer {label:?} requests {requested} bytes, exceeding device max_storage_buffer_binding_size {maximum}"
            ),
        }
    }
}

impl std::error::Error for GpuStartupAllocationError {}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AdapterMetadata {
    name: String,
    vendor: u32,
    device: u32,
    device_type: wgpu::DeviceType,
    driver: String,
    driver_info: String,
    backend: wgpu::Backend,
}

/// Immutable identity of the adapter selected by the authoritative GPU
/// backend. This is evidence about the backend that executes work; it does not
/// enumerate or rediscover a second adapter for reporting.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuDeviceIdentity {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: String,
    pub wgpu_backend: String,
    pub driver: String,
    pub driver_info: String,
    pub compute_plane: String,
    pub software_adapter: bool,
}

impl AdapterMetadata {
    fn identity(&self, compute_plane: &str) -> GpuDeviceIdentity {
        GpuDeviceIdentity {
            name: self.name.clone(),
            vendor_id: self.vendor,
            device_id: self.device,
            device_type: format!("{:?}", self.device_type),
            wgpu_backend: self.backend.to_str().to_string(),
            driver: self.driver.clone(),
            driver_info: self.driver_info.clone(),
            compute_plane: compute_plane.to_string(),
            software_adapter: self.is_software(),
        }
    }
}

impl AdapterMetadata {
    fn from_info(info: wgpu::AdapterInfo) -> Self {
        Self {
            name: info.name,
            vendor: info.vendor,
            device: info.device,
            device_type: info.device_type,
            driver: info.driver,
            driver_info: info.driver_info,
            backend: info.backend,
        }
    }

    fn is_software(&self) -> bool {
        if self.device_type == wgpu::DeviceType::Cpu {
            return true;
        }

        let text = format!(
            "{} {} {}",
            self.name, self.driver, self.driver_info
        )
        .to_ascii_lowercase();
        [
            "llvmpipe",
            "lavapipe",
            "softpipe",
            "swrast",
            "openswr",
            "swiftshader",
            "software",
            "warp",
        ]
            .iter()
            .any(|needle| text.contains(needle))
    }

    fn is_non_cpu_gpu(&self) -> bool {
        !self.is_software()
    }

    fn matches(&self, other: &Self) -> bool {
        self.name == other.name
            && self.vendor == other.vendor
            && self.device == other.device
            && self.device_type == other.device_type
            && self.backend == other.backend
    }

    fn summary(&self) -> String {
        format!(
            "{} via {} ({:?}, vendor={:#06x}, device={:#06x}, driver={})",
            self.name, self.backend, self.device_type, self.vendor, self.device, self.driver
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AdapterSelectionError {
    NoAdapters,
    OnlySoftware { count: usize },
}

impl fmt::Display for AdapterSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdapterSelectionError::NoAdapters => {
                write!(f, "no adapters exposed by wgpu")
            }
            AdapterSelectionError::OnlySoftware { count } => {
                write!(f, "only software adapters found by wgpu ({count})")
            }
        }
    }
}

fn allow_software_wgpu_adapter() -> bool {
    std::env::var(ALLOW_SOFTWARE_WGPU_ADAPTER_ENV)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn required_wgpu_features() -> wgpu::Features {
    wgpu::Features::PUSH_CONSTANTS
}

fn required_wgpu_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_push_constant_size: 32,
        ..wgpu::Limits::default()
    }
}

fn wgpu_compute_plane(backend: wgpu::Backend) -> String {
    format!("wgpu-{}", backend.to_str())
}

fn select_wgpu_adapter_candidates(
    adapters: &[AdapterMetadata],
    high_performance_index: Option<usize>,
    allow_software: bool,
) -> std::result::Result<Vec<usize>, AdapterSelectionError> {
    if adapters.is_empty() {
        return Err(AdapterSelectionError::NoAdapters);
    }

    let mut selected = Vec::with_capacity(adapters.len());
    let mut push_unique = |idx: usize| {
        if !selected.contains(&idx) {
            selected.push(idx);
        }
    };

    if let Some(idx) = high_performance_index.filter(|idx| *idx < adapters.len()) {
        if allow_software || adapters[idx].is_non_cpu_gpu() {
            push_unique(idx);
        }
    }

    for (idx, meta) in adapters.iter().enumerate() {
        if meta.device_type == wgpu::DeviceType::DiscreteGpu && meta.is_non_cpu_gpu() {
            push_unique(idx);
        }
    }

    for (idx, meta) in adapters.iter().enumerate() {
        if meta.is_non_cpu_gpu() {
            push_unique(idx);
        }
    }

    if allow_software {
        for idx in 0..adapters.len() {
            push_unique(idx);
        }
    }

    if selected.is_empty() {
        Err(AdapterSelectionError::OnlySoftware {
            count: adapters.len(),
        })
    } else {
        Ok(selected)
    }
}

struct EnumeratedAdapter {
    adapter: wgpu::Adapter,
    metadata: AdapterMetadata,
}

/// Zero-copy view of a f16 tensor borrowed from the caller.
#[derive(Copy, Clone, Debug)]
pub struct TensorView<'a> {
    pub data: &'a [half::f16],
    pub rows: usize,
    pub cols: usize,
}

/// Zero-copy mutable view of a f16 tensor borrowed from the caller.
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    pub data: &'a mut [half::f16],
    pub rows: usize,
    pub cols: usize,
}

/// Failure stage for one routed-expert GPU activation.
///
/// `wgpu` 0.20 exposes buffer-map results and a device-loss callback, but
/// `Queue::submit` itself returns only a submission index. Validation and
/// submission therefore remain typed boundaries that hardware-independent
/// tests can inject; production does not claim synchronous detection where
/// the API cannot attribute an error to one concurrent expert dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuExpertDispatchErrorKind {
    ResidencyMiss,
    PhysicalCapacity,
    Upload,
    #[allow(dead_code)] // wgpu 0.20 has no per-dispatch synchronous validation result.
    ValidationDispatch,
    #[allow(dead_code)] // Queue::submit returns only a SubmissionIndex in wgpu 0.20.
    Submission,
    ReadbackChannel,
    ReadbackMap,
    DeviceLost,
    RuntimeInvariant,
}

/// Fail-closed error category for the crate-private raw Q4_0 qualification
/// dispatch. These errors are deliberately separate from serving dispatch
/// errors because this seam neither admits nor executes a routed expert.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Q4ParityGpuErrorKind {
    InvalidRequest,
    ResourceUnavailable,
    ResourceCreation,
    Validation,
    ReadbackTimeout,
    ReadbackChannel,
    ReadbackMap,
    DeviceLost,
    NonfiniteOutput,
}

/// Typed raw-Q4 qualification failure returned only within this crate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Q4ParityGpuError {
    pub kind: Q4ParityGpuErrorKind,
    pub detail: String,
}

impl Q4ParityGpuError {
    fn new(kind: Q4ParityGpuErrorKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for Q4ParityGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.detail)
    }
}

impl std::error::Error for Q4ParityGpuError {}

impl fmt::Display for GpuExpertDispatchErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::ResidencyMiss => "residency-miss",
            Self::PhysicalCapacity => "physical-capacity",
            Self::Upload => "upload",
            Self::ValidationDispatch => "validation-dispatch",
            Self::Submission => "submission",
            Self::ReadbackChannel => "readback-channel",
            Self::ReadbackMap => "readback-map",
            Self::DeviceLost => "device-lost",
            Self::RuntimeInvariant => "runtime-invariant",
        };
        f.write_str(name)
    }
}

/// Typed boundary between GPU routed-expert execution and the engine's
/// strict-vs-serving recovery policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuExpertDispatchError {
    pub layer: u32,
    pub expert_id: u32,
    pub kind: GpuExpertDispatchErrorKind,
    pub detail: String,
}

impl GpuExpertDispatchError {
    pub(crate) fn new(
        layer: u32,
        expert_id: u32,
        kind: GpuExpertDispatchErrorKind,
        detail: impl Into<String>,
    ) -> Self {
        Self { layer, expert_id, kind, detail: detail.into() }
    }
}

impl fmt::Display for GpuExpertDispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPU routed expert {} (layer {}) failed at {}: {}",
            self.expert_id, self.layer, self.kind, self.detail
        )
    }
}

impl std::error::Error for GpuExpertDispatchError {}

/// PR4's exact MER-owned routed-expert GPU memory snapshot. Its scope is
/// deliberately narrower than process/device memory: physical expert weight
/// buffers plus the fixed routed-expert workspace buffers. Dense, attention,
/// KV, driver, and allocator overhead remain outside this internal ledger.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[allow(dead_code)] // Public PR5 seam; PR4 validates it through backend-local tests.
pub struct GpuExpertMemorySnapshot {
    /// Host payload bytes admitted by [`crate::expert_cache::GpuExpertCache`].
    pub logical_admitted_bytes: u64,
    /// All live physical expert weight buffers, including evicted entries
    /// retained by an in-flight `Arc`.
    pub expert_live_bytes: u64,
    /// Physical expert bytes still addressable through the registry.
    pub expert_registry_bytes: u64,
    /// Fixed device buffers owned by the routed-expert workspace pool.
    pub workspace_bytes: u64,
    /// `expert_live_bytes + workspace_bytes`.
    pub total_tracked_bytes: u64,
    /// Configured capacity for physical expert weight buffers only.
    pub expert_capacity_bytes: u64,
    pub physical_entries: usize,
    pub physical_installs: u64,
    pub physical_evictions: u64,
    pub stale_retirements: u64,
}

/// Non-mutating identity evidence for one physically resident expert. This is
/// exposed only through the crate-private numerical-qualification seam; it
/// neither touches physical LRU recency nor participates in serving dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct GpuPhysicalExpertResidency {
    pub(crate) expert_id: u32,
    pub(crate) generation: u64,
    pub(crate) device_bytes: u64,
}

/// Monotonic counters for only the routed-expert GPU path. Dense and
/// attention operations are deliberately excluded.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuExpertIoSnapshot {
    /// Actual physical expert-buffer writes; registry hits do not increment it.
    pub expert_weight_uploads: u64,
    /// Exact bytes passed to successful expert-buffer write calls.
    pub expert_weight_upload_bytes: u64,
    /// Actual routed hidden-state workspace writes.
    pub hidden_state_uploads: u64,
    /// Exact bytes passed to routed hidden-state workspace writes.
    pub hidden_state_upload_bytes: u64,
    /// Routed command-buffer submissions. A returned `SubmissionIndex` is not
    /// represented as synchronous wgpu validation success.
    pub queue_submissions: u64,
    /// Routed staging-buffer `map_async` requests.
    pub map_requests: u64,
    /// Successfully mapped and copied routed readbacks.
    pub readback_completions: u64,
    /// Exact output bytes copied by completed routed readbacks.
    pub readback_bytes: u64,
}

#[derive(Default)]
struct GpuExpertIoCounters {
    expert_weight_uploads: AtomicU64,
    expert_weight_upload_bytes: AtomicU64,
    hidden_state_uploads: AtomicU64,
    hidden_state_upload_bytes: AtomicU64,
    queue_submissions: AtomicU64,
    map_requests: AtomicU64,
    readback_completions: AtomicU64,
    readback_bytes: AtomicU64,
}

impl GpuExpertIoCounters {
    fn snapshot(&self) -> GpuExpertIoSnapshot {
        GpuExpertIoSnapshot {
            expert_weight_uploads: self.expert_weight_uploads.load(Ordering::Relaxed),
            expert_weight_upload_bytes: self
                .expert_weight_upload_bytes
                .load(Ordering::Relaxed),
            hidden_state_uploads: self.hidden_state_uploads.load(Ordering::Relaxed),
            hidden_state_upload_bytes: self.hidden_state_upload_bytes.load(Ordering::Relaxed),
            queue_submissions: self.queue_submissions.load(Ordering::Relaxed),
            map_requests: self.map_requests.load(Ordering::Relaxed),
            readback_completions: self.readback_completions.load(Ordering::Relaxed),
            readback_bytes: self.readback_bytes.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PhysicalExpertKey {
    expert_id: u32,
    generation: u64,
}

trait PhysicalRegistryEntry: Send + Sync + 'static {
    fn physical_key(&self) -> PhysicalExpertKey;
    fn device_bytes(&self) -> u64;
}

#[allow(dead_code)] // `workspace_bytes` is consumed through the PR5 snapshot seam.
struct GpuMemoryLedger {
    expert_live_bytes: AtomicU64,
    workspace_bytes: u64,
    expert_capacity_bytes: u64,
}

impl GpuMemoryLedger {
    fn new(workspace_bytes: u64, expert_capacity_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            expert_live_bytes: AtomicU64::new(0),
            workspace_bytes,
            expert_capacity_bytes,
        })
    }

    fn expert_live_bytes(&self) -> u64 {
        self.expert_live_bytes.load(Ordering::Acquire)
    }

    fn acquire(self: &Arc<Self>, bytes: u64) -> std::result::Result<ExpertAllocationLease, String> {
        if bytes == 0 {
            return Err("physical expert allocation must be non-zero".to_string());
        }
        self.expert_live_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                live.checked_add(bytes)
                    .filter(|next| *next <= self.expert_capacity_bytes)
            })
            .map_err(|live| {
                format!(
                    "physical expert live-byte ledger capacity/overflow: capacity={} live={live} add={bytes}",
                    self.expert_capacity_bytes
                )
            })?;
        Ok(ExpertAllocationLease {
            ledger: self.clone(),
            bytes,
        })
    }
}

/// Non-cloneable charge owned by one physical allocation. An entry may be
/// removed from the registry while in flight; only the final entry `Arc` drop
/// releases this lease and decrements live bytes.
struct ExpertAllocationLease {
    ledger: Arc<GpuMemoryLedger>,
    bytes: u64,
}

impl Drop for ExpertAllocationLease {
    fn drop(&mut self) {
        self.ledger
            .expert_live_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                live.checked_sub(self.bytes)
            })
            .expect("physical expert live-byte ledger underflow");
    }
}

#[derive(Clone, Copy)]
struct InstallReservation {
    bytes: u64,
    charged: bool,
}

struct PhysicalRegistryInner<T: PhysicalRegistryEntry> {
    entries: lru::LruCache<u32, Arc<T>>,
    registry_bytes: u64,
    reserved_bytes: u64,
    installing: HashMap<PhysicalExpertKey, InstallReservation>,
    installs: u64,
    evictions: u64,
    stale_retirements: u64,
}

struct PhysicalGpuExpertRegistry<T: PhysicalRegistryEntry> {
    inner: ParkingMutex<PhysicalRegistryInner<T>>,
    install_cv: parking_lot::Condvar,
    ledger: Arc<GpuMemoryLedger>,
}

#[derive(Debug)]
struct PhysicalCapacityError {
    requested_bytes: u64,
    expert_capacity_bytes: u64,
    expert_live_bytes: u64,
    expert_registry_bytes: u64,
    reserved_bytes: u64,
}

impl fmt::Display for PhysicalCapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "physical expert allocation of {} bytes exceeds available capacity: capacity={} live={} registry={} reserved={}",
            self.requested_bytes,
            self.expert_capacity_bytes,
            self.expert_live_bytes,
            self.expert_registry_bytes,
            self.reserved_bytes
        )
    }
}

enum PhysicalRegistryAcquire<T: PhysicalRegistryEntry> {
    Hit(Arc<T>),
    Install(PhysicalInstallPermit<T>),
    StaleRequester,
}

/// Directional generation result for one expert-id lookup. A newer request
/// may retire an older stored generation; an older request must never disturb
/// a newer stored generation.
enum PhysicalRegistryLookup<T: PhysicalRegistryEntry> {
    Hit(Arc<T>),
    Miss,
    StaleRequester,
}

struct PhysicalInstallPermit<T: PhysicalRegistryEntry> {
    registry: Arc<PhysicalGpuExpertRegistry<T>>,
    key: PhysicalExpertKey,
    bytes: u64,
    active: bool,
    charged: bool,
}

impl<T: PhysicalRegistryEntry> PhysicalGpuExpertRegistry<T> {
    fn new(ledger: Arc<GpuMemoryLedger>) -> Arc<Self> {
        Arc::new(Self {
            inner: ParkingMutex::new(PhysicalRegistryInner {
                entries: lru::LruCache::unbounded(),
                registry_bytes: 0,
                reserved_bytes: 0,
                installing: HashMap::new(),
                installs: 0,
                evictions: 0,
                stale_retirements: 0,
            }),
            install_cv: parking_lot::Condvar::new(),
            ledger,
        })
    }

    fn capacity_bytes(&self) -> u64 {
        self.ledger.expert_capacity_bytes
    }

    /// Return an addressable physical entry only when its logical generation
    /// still matches. This is the routed-expert fast path: it performs one
    /// registry lock and no upload planning or host-weight copy.
    fn lookup_current(&self, key: PhysicalExpertKey) -> PhysicalRegistryLookup<T> {
        let mut inner = self.inner.lock();
        Self::lookup_current_locked(&mut inner, key)
    }

    fn lookup_current_locked(
        inner: &mut PhysicalRegistryInner<T>,
        key: PhysicalExpertKey,
    ) -> PhysicalRegistryLookup<T> {
        let Some(existing) = inner.entries.peek(&key.expert_id).cloned() else {
            return PhysicalRegistryLookup::Miss;
        };
        let stored_key = existing.physical_key();
        debug_assert_eq!(stored_key.expert_id, key.expert_id);
        if stored_key.generation == key.generation {
            return PhysicalRegistryLookup::Hit(
                inner
                    .entries
                    .get(&key.expert_id)
                    .cloned()
                    .expect("validated physical entry must remain present under lock"),
            );
        }
        if stored_key.generation > key.generation {
            return PhysicalRegistryLookup::StaleRequester;
        }

        Self::retire_entry_locked(inner, key.expert_id);
        PhysicalRegistryLookup::Miss
    }

    fn retire_entry_locked(inner: &mut PhysicalRegistryInner<T>, expert_id: u32) {
        let stale = inner
            .entries
            .pop(&expert_id)
            .expect("validated physical entry must remain present under lock");
        inner.registry_bytes = inner
            .registry_bytes
            .checked_sub(stale.device_bytes())
            .expect("physical registry byte underflow retiring stale entry");
        inner.stale_retirements = inner.stale_retirements.saturating_add(1);
        drop(stale);
    }

    /// Validate `(id, generation)` and either return the current physical
    /// entry or reserve exact capacity for one keyed install. Concurrent
    /// demand for the same id waits on the keyed in-progress state and then
    /// reuses the winner; the registry lock is never held during upload.
    fn acquire_or_reserve(
        self: &Arc<Self>,
        key: PhysicalExpertKey,
        bytes: u64,
    ) -> std::result::Result<PhysicalRegistryAcquire<T>, PhysicalCapacityError> {
        let capacity = self.capacity_bytes();
        if bytes == 0 || bytes > capacity {
            return Err(self.capacity_error(bytes, 0, 0));
        }

        let mut inner = self.inner.lock();
        loop {
            match Self::lookup_current_locked(&mut inner, key) {
                PhysicalRegistryLookup::Hit(hit) => {
                    return Ok(PhysicalRegistryAcquire::Hit(hit));
                }
                PhysicalRegistryLookup::Miss => {}
                PhysicalRegistryLookup::StaleRequester => {
                    return Ok(PhysicalRegistryAcquire::StaleRequester);
                }
            }

            if inner
                .installing
                .keys()
                .any(|installing| installing.expert_id == key.expert_id)
            {
                self.install_cv.wait(&mut inner);
                continue;
            }

            while !self.install_fits(&inner, bytes) {
                let Some((_, victim)) = inner.entries.pop_lru() else {
                    return Err(self.capacity_error(
                        bytes,
                        inner.registry_bytes,
                        inner.reserved_bytes,
                    ));
                };
                inner.registry_bytes = inner
                    .registry_bytes
                    .checked_sub(victim.device_bytes())
                    .expect("physical registry byte underflow during eviction");
                inner.evictions = inner.evictions.saturating_add(1);
                // Release registry ownership before re-reading live bytes. If
                // an activation still owns the Arc, the RAII lease remains.
                drop(victim);
            }

            inner.reserved_bytes = inner
                .reserved_bytes
                .checked_add(bytes)
                .expect("physical install reservation byte overflow");
            let previous = inner.installing.insert(
                key,
                InstallReservation {
                    bytes,
                    charged: false,
                },
            );
            debug_assert!(previous.is_none());
            return Ok(PhysicalRegistryAcquire::Install(PhysicalInstallPermit {
                registry: self.clone(),
                key,
                bytes,
                active: true,
                charged: false,
            }));
        }
    }

    fn install_fits(&self, inner: &PhysicalRegistryInner<T>, bytes: u64) -> bool {
        inner
            .registry_bytes
            .checked_add(inner.reserved_bytes)
            .and_then(|used| used.checked_add(bytes))
            .is_some_and(|used| used <= self.capacity_bytes())
            && self
                .ledger
                .expert_live_bytes()
                .checked_add(inner.reserved_bytes)
                .and_then(|used| used.checked_add(bytes))
                .is_some_and(|used| used <= self.capacity_bytes())
    }

    fn capacity_error(
        &self,
        requested_bytes: u64,
        expert_registry_bytes: u64,
        reserved_bytes: u64,
    ) -> PhysicalCapacityError {
        PhysicalCapacityError {
            requested_bytes,
            expert_capacity_bytes: self.capacity_bytes(),
            expert_live_bytes: self.ledger.expert_live_bytes(),
            expert_registry_bytes,
            reserved_bytes,
        }
    }

    /// Logical miss retirement is O(1): remove only this id. Unrelated stale
    /// entries remain accounted and capacity-evictable, but cannot execute
    /// because every routed lookup starts with logical admission validation.
    fn retire_logical_miss(&self, expert_id: u32) {
        let mut inner = self.inner.lock();
        if inner.entries.peek(&expert_id).is_some() {
            Self::retire_entry_locked(&mut inner, expert_id);
        }
    }

    /// Retire only the stale allocation this dispatch observed. If another
    /// thread has already installed a newer generation, leave it addressable.
    fn retire_key_if_present(&self, key: PhysicalExpertKey) {
        let mut inner = self.inner.lock();
        if inner
            .entries
            .peek(&key.expert_id)
            .is_some_and(|entry| entry.physical_key() == key)
        {
            Self::retire_entry_locked(&mut inner, key.expert_id);
        }
    }

    fn snapshot(&self, logical_admitted_bytes: u64) -> GpuExpertMemorySnapshot {
        let inner = self.inner.lock();
        let expert_live_bytes = self.ledger.expert_live_bytes();
        let workspace_bytes = self.ledger.workspace_bytes;
        GpuExpertMemorySnapshot {
            logical_admitted_bytes,
            expert_live_bytes,
            expert_registry_bytes: inner.registry_bytes,
            workspace_bytes,
            total_tracked_bytes: expert_live_bytes
                .checked_add(workspace_bytes)
                .expect("tracked GPU byte total overflow"),
            expert_capacity_bytes: self.capacity_bytes(),
            physical_entries: inner.entries.len(),
            physical_installs: inner.installs,
            physical_evictions: inner.evictions,
            stale_retirements: inner.stale_retirements,
        }
    }

    fn residency_evidence(&self, expert_id: u32) -> Option<GpuPhysicalExpertResidency> {
        let inner = self.inner.lock();
        inner.entries.peek(&expert_id).map(|entry| {
            let key = entry.physical_key();
            GpuPhysicalExpertResidency {
                expert_id: key.expert_id,
                generation: key.generation,
                device_bytes: entry.device_bytes(),
            }
        })
    }

    #[cfg(test)]
    fn contains_key(&self, key: PhysicalExpertKey) -> bool {
        self.inner
            .lock()
            .entries
            .peek(&key.expert_id)
            .is_some_and(|entry| entry.physical_key() == key)
    }
}

impl<T: PhysicalRegistryEntry> PhysicalInstallPermit<T> {
    /// Transfer this permit's capacity reservation into an exact live-byte
    /// lease immediately after the device allocation is successfully created.
    fn charge_allocation(
        &mut self,
    ) -> std::result::Result<ExpertAllocationLease, String> {
        if !self.active || self.charged {
            return Err("physical install permit is not chargeable".to_string());
        }
        let mut inner = self.registry.inner.lock();
        let reservation = *inner
            .installing
            .get(&self.key)
            .ok_or_else(|| "physical install reservation disappeared".to_string())?;
        if reservation.charged || reservation.bytes != self.bytes {
            return Err("physical install reservation state mismatch".to_string());
        }
        let lease = self.registry.ledger.acquire(self.bytes)?;
        inner.reserved_bytes = inner
            .reserved_bytes
            .checked_sub(self.bytes)
            .ok_or_else(|| "physical install reserved-byte underflow".to_string())?;
        inner
            .installing
            .get_mut(&self.key)
            .expect("validated physical install reservation disappeared under lock")
            .charged = true;
        self.charged = true;
        Ok(lease)
    }

    fn install(mut self, entry: Arc<T>) -> std::result::Result<Arc<T>, String> {
        if !self.active || !self.charged {
            return Err("physical install permit was not charged".to_string());
        }
        if entry.physical_key() != self.key || entry.device_bytes() != self.bytes {
            return Err("physical install entry does not match its reservation".to_string());
        }
        let mut inner = self.registry.inner.lock();
        // On every error below, this local guard must drop before the by-value
        // permit argument runs `Drop` and re-locks the same registry.
        let reservation = *inner
            .installing
            .get(&self.key)
            .ok_or_else(|| "physical install reservation disappeared".to_string())?;
        if !reservation.charged || reservation.bytes != self.bytes {
            return Err("physical install reservation state mismatch".to_string());
        }
        if inner.entries.peek(&self.key.expert_id).is_some() {
            return Err("physical entry appeared while its keyed install was reserved".to_string());
        }
        let new_registry_bytes = inner
            .registry_bytes
            .checked_add(self.bytes)
            .ok_or_else(|| "physical registry byte overflow".to_string())?;
        if new_registry_bytes > self.registry.capacity_bytes() {
            return Err("physical registry exceeded configured capacity".to_string());
        }
        inner.installing.remove(&self.key);
        let previous = inner.entries.put(self.key.expert_id, entry.clone());
        debug_assert!(previous.is_none());
        inner.registry_bytes = new_registry_bytes;
        inner.installs = inner.installs.saturating_add(1);
        self.active = false;
        drop(inner);
        self.registry.install_cv.notify_all();
        Ok(entry)
    }
}

impl<T: PhysicalRegistryEntry> Drop for PhysicalInstallPermit<T> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut inner = self.registry.inner.lock();
        if let Some(reservation) = inner.installing.remove(&self.key) {
            if !reservation.charged {
                inner.reserved_bytes = inner
                    .reserved_bytes
                    .checked_sub(reservation.bytes)
                    .expect("physical install reserved-byte underflow on cancellation");
            }
        }
        self.active = false;
        drop(inner);
        self.registry.install_cv.notify_all();
    }
}

/// Abstraction over GPU-resident storage for expert weight buffers.
///
/// `GpuResident` (host bytes) implements this returning `None` for
/// `as_wgpu_buffer`.  `VramExpertEntry` (fully promoted) returns `Some`.
/// A future CUDA backend would add a third implementor wrapping a device
/// pointer opaquely without leaking it here.
pub trait GpuStorage: Send + Sync + 'static {
    /// Total byte length of the weight payload.
    fn byte_len(&self) -> usize;
    /// VRAM buffer handle, if this storage is device-resident.
    /// Returns `None` for host-only (CPU-tier) storage.
    fn as_wgpu_buffer(&self) -> Option<&wgpu::Buffer>;
}

/// Minimal contract every math backend must satisfy.
pub trait Backend: Send + Sync + 'static {
    fn device_name(&self) -> &str;
    fn is_gpu(&self) -> bool {
        false
    }
    fn matmul_into(&self, a: TensorView, b: TensorView, out: &mut TensorViewMut) -> Result<()>;
    fn swiglu_into(&self, gate: TensorView, up: TensorView, out: &mut TensorViewMut) -> Result<()>;
    fn softmax(&self, x: &mut TensorViewMut) -> Result<()>;
    fn kv_cache_insert(
        &self,
        layer: usize,
        position: usize,
        k: TensorView,
        v: TensorView,
    ) -> Result<()>;
    fn kv_attend(
        &self,
        layer: usize,
        q: TensorView,
        seq_len: usize,
        out: &mut TensorViewMut,
    ) -> Result<()>;

    /// Execute one MoE expert FFN from VRAM when the expert is GPU-resident,
    /// or fall back to the CPU path. On the GPU path the weight bytes are
    /// already in VRAM and no PCIe upload is needed.
    ///
    /// `x`       : hidden state input  [d_model]
    /// `d_model` : hidden dimension
    /// `d_ff`    : FFN intermediate dimension
    /// `out`     : output buffer        [d_model]
    fn expert_matmul(
        &self,
        layer_idx: usize,
        expert_id: u32,
        x:        TensorView<'_>,
        d_model:  usize,
        d_ff:     usize,
        out:      &mut TensorViewMut<'_>,
    ) -> Result<()>;
}

// =====================================================================
// Push Constants structs (POD, 16 bytes max, byte-identical to WGSL)
// =====================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulPushConstants {
    m: u32,
    n: u32,
    k: u32,
    /// Unused (zero) for the dense F32 `matmul_main` entry point. For
    /// the Q4_0 inline-dequant entry point (`matmul_q4_0_main`) this
    /// carries the projection's first-block index inside the packed
    /// expert weight buffer — see `wgpu_shaders/matmul_q4_0.wgsl`.
    w_block_off: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SwigluPushConstants {
    n_elements: u32,
    /// GPT-OSS SwiGLU gate clamp threshold. `+inf` disables the clamp
    /// (`clamp(g, -inf, inf)` is a bit-exact no-op), matching the CPU path.
    swiglu_limit: f32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxPushConstants {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AttentionPushConstants {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    /// Offset of this layer's K slice in the KV buffer, in **f32
    /// elements** (not bytes — a byte offset would overflow u32 for
    /// deep models with large KV slices).
    layer_offset: u32,
    /// Value head dimension (Finding 12). Equal to `head_dim` for symmetric
    /// architectures; differs for asymmetric-V models. Drives the V slice
    /// stride (`num_kv_heads * v_head_dim`) and the attention-output width
    /// (`num_heads * v_head_dim`) independently of the Q/K `head_dim`.
    v_head_dim: u32,
    _pad0: u32,
    _pad1: u32,
}

// =====================================================================
// GPU VRAM KV Cache
// =====================================================================

/// Number of f32 elements occupied by one layer's `[K | V]` region in the
/// packed KV buffer: `max_seq_len * (k_dim + v_dim)`. Pure arithmetic split
/// out from [`GpuKvCache`] so the asymmetric-V layout (Finding 12) can be
/// unit-tested without a live `wgpu::Buffer`.
#[inline]
pub fn kv_layer_stride_elems(max_seq_len: usize, k_dim: usize, v_dim: usize) -> usize {
    max_seq_len * (k_dim + v_dim)
}

/// f32-element index of a `(layer, kv, seq_pos)` slot within the packed KV
/// buffer. `kv == 0` selects the K region (stride `k_dim`); any other value
/// selects the V region (stride `v_dim`), which begins after the whole K
/// region of the same layer. The K and V widths are addressed independently
/// so asymmetric value widths (`v_dim != k_dim`, Finding 12) stay correct.
#[inline]
pub fn kv_offset_elems(
    layer: usize,
    kv: usize,
    seq_pos: usize,
    max_seq_len: usize,
    k_dim: usize,
    v_dim: usize,
) -> usize {
    let layer_base = layer * kv_layer_stride_elems(max_seq_len, k_dim, v_dim);
    if kv == 0 {
        layer_base + seq_pos * k_dim
    } else {
        layer_base + max_seq_len * k_dim + seq_pos * v_dim
    }
}

pub struct GpuKvCache {
    pub buffer: wgpu::Buffer,
    pub num_layers: usize,
    pub max_seq_len: usize,
    /// K slice width per position: `num_kv_heads * head_dim`.
    pub k_dim: usize,
    /// V slice width per position: `num_kv_heads * v_head_dim`. Equal to
    /// `k_dim` for every symmetric architecture; larger or smaller when the
    /// value head dimension differs from the query/key head dimension
    /// (Finding 12, e.g. MiMo-V2-Flash `v_head_dim != head_dim`).
    pub v_dim: usize,
}

impl GpuKvCache {
    /// Number of f32 elements occupied by one layer's `[K | V]` region:
    /// `max_seq_len * (k_dim + v_dim)`. The K sub-region occupies the first
    /// `max_seq_len * k_dim` elements and the V sub-region follows it.
    #[inline]
    pub fn layer_stride_elems(&self) -> usize {
        kv_layer_stride_elems(self.max_seq_len, self.k_dim, self.v_dim)
    }

    /// Byte offset of a `(layer, kv, seq_pos)` slot within the KV buffer.
    /// `kv == 0` selects the K region (stride `k_dim`); `kv == 1` selects the
    /// V region (stride `v_dim`), which begins after the whole K region. The
    /// asymmetric K/V widths make a single shared stride incorrect, so the
    /// two regions are addressed independently (Finding 12).
    pub fn offset_bytes(&self, layer: usize, kv: usize, seq_pos: usize) -> u64 {
        let idx = kv_offset_elems(
            layer,
            kv,
            seq_pos,
            self.max_seq_len,
            self.k_dim,
            self.v_dim,
        );
        (idx * 4) as u64
    }
}

// =====================================================================
// GPU Backend using wgpu
// =====================================================================

/// How the bytes inside a [`VramExpertEntry`] weight buffer are encoded,
/// and therefore which matmul pipeline the FFN passes must dispatch.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum VramWeightLayout {
    /// Dense little-endian f32: `[gate | up | down]`, each projection
    /// `d_ff × d_model × 4` bytes. Bind groups slice the buffer per
    /// projection; `matmul_main` consumes it.
    F32,
    /// Native GGUF Q4_0 blocks (18 bytes / 32 weights), gate, up and
    /// down concatenated back-to-back with no padding. The whole
    /// buffer is bound at offset 0 (18-byte blocks cannot honour
    /// storage-offset alignment), and each pass selects its projection
    /// via the `w_block_off` push constant; `matmul_q4_0_main`
    /// dequantises inline.
    Q4_0,
}

#[derive(Clone, Copy)]
struct F32ExpertUploadPlan {
    device_bytes: u64,
    projection_offset: u64,
}

#[derive(Clone, Copy)]
struct Q4ExpertUploadPlan {
    device_bytes: u64,
    required_bytes: usize,
    up_block_offset: u32,
    down_block_offset: u32,
}

#[derive(Clone, Copy)]
enum ExpertUploadPlan {
    F32(F32ExpertUploadPlan),
    Q4_0(Q4ExpertUploadPlan),
}

impl ExpertUploadPlan {
    fn device_bytes(self) -> u64 {
        match self {
            Self::F32(plan) => plan.device_bytes,
            Self::Q4_0(plan) => plan.device_bytes,
        }
    }
}

/// A fully-initialized VRAM expert: weight buffer + shape/layout
/// metadata for the four FFN dispatch passes (the dispatch-time bind
/// groups are built per call against the checked-out
/// [`ExpertWorkspace`]). Created once per expert on first promotion;
/// reused on every subsequent token.
struct VramExpertEntry {
    key: PhysicalExpertKey,
    /// Raw weight buffer in VRAM. Layout: [gate_proj | up_proj | down_proj],
    /// either dense f32 LE or packed Q4_0 blocks — see [`VramWeightLayout`].
    /// gate_proj: [d_ff, d_model], up_proj: [d_ff, d_model], down_proj: [d_model, d_ff].
    weight_buf: wgpu::Buffer,
    /// Cached shape parameters.
    d_model:   usize,
    d_ff:      usize,
    /// Weight encoding → which matmul pipeline the passes dispatch.
    layout:    VramWeightLayout,
    /// Bytes per projection matrix (F32 layout only; selects the
    /// gate/up/down sub-range of `weight_buf` when the per-dispatch
    /// bind groups are built — gate at 0, up at `proj_bytes`, down at
    /// `2 * proj_bytes`). Unused (0) for Q4_0, whose projection base
    /// travels in the `w_block_off` push constant instead.
    proj_bytes: u64,
    /// First-block index of the up projection (Q4_0 layout only; 0 for F32).
    up_block_off:   u32,
    /// First-block index of the down projection (Q4_0 layout only; 0 for F32).
    down_block_off: u32,
    /// Exact descriptor size of `weight_buf`.
    device_bytes: u64,
    /// Final-Arc live-byte accounting charge for `weight_buf`.
    _allocation: ExpertAllocationLease,
}

impl PhysicalRegistryEntry for VramExpertEntry {
    fn physical_key(&self) -> PhysicalExpertKey {
        self.key
    }

    fn device_bytes(&self) -> u64 {
        self.device_bytes
    }
}

impl GpuStorage for VramExpertEntry {
    fn byte_len(&self) -> usize {
        self.device_bytes as usize
    }
    fn as_wgpu_buffer(&self) -> Option<&wgpu::Buffer> {
        Some(&self.weight_buf)
    }
}

/// Number of per-dispatch expert FFN workspaces pre-allocated at
/// backend init. Each VRAM expert dispatch checks one out for its
/// lifetime, so up to this many expert FFNs can be in flight on the
/// queue **concurrently** — the per-dispatch wait below only blocks
/// on its own submission index, never on the whole queue. Sized at
/// five device buffers per workspace. PR4 derives their exact descriptor
/// bytes and excludes the host scratch vector from the GPU ledger.
const EXPERT_WORKSPACE_POOL: usize = 4;
const EXPERT_WORKSPACE_DEVICE_BUFFERS: u64 = 5;

fn expert_workspace_device_bytes(buffer_bytes: u64, workspace_count: usize) -> Option<u64> {
    let workspace_count = u64::try_from(workspace_count).ok()?;
    buffer_bytes
        .checked_mul(EXPERT_WORKSPACE_DEVICE_BUFFERS)
        .and_then(|per_workspace| per_workspace.checked_mul(workspace_count))
}

/// Private buffer set for one in-flight expert FFN dispatch.
///
/// The legacy path funnelled every expert through the backend-global
/// `work_a` / `work_mid_*` / `staging_dn` buffers, which forced the
/// whole FFN (upload → 4 passes → readback) under one
/// `expert_execution_lock` — and the per-op `Maintain::Wait` then
/// stalled the *entire* device queue per dispatch. Giving each
/// dispatch its own buffers removes both: no shared-buffer lock, and
/// each dispatch waits only for its own `SubmissionIndex`.
///
/// All buffers are sized for the worst-case expert shape
/// (`MAX_EXPERT_D_FF` f32 elements — `d_model ≤ MAX_EXPERT_D_FF` was
/// already implied by the legacy path, which wrote the [d_model] down
/// output into the `MAX_EXPERT_D_FF`-sized `work_mid_1`).
struct ExpertWorkspace {
    /// Hidden-state upload target ([d_model] f32).
    x_buf:   wgpu::Buffer,
    /// Gate projection output, then reused for the final down output ([d_ff] / [d_model] f32).
    mid_1:   wgpu::Buffer,
    /// Up projection output ([d_ff] f32).
    mid_2:   wgpu::Buffer,
    /// SwiGLU output ([d_ff] f32).
    ffn_out: wgpu::Buffer,
    /// Readback staging for the down output ([d_model] f32, MAP_READ).
    staging: wgpu::Buffer,
    /// Host-side f16→f32 conversion scratch for the `x` upload —
    /// per-workspace, so expert dispatches never contend on the
    /// backend-global `conversion_scratch` either.
    scratch: Vec<f32>,
}

impl ExpertWorkspace {
    fn device_bytes(&self) -> u64 {
        self.x_buf
            .size()
            .checked_add(self.mid_1.size())
            .and_then(|bytes| bytes.checked_add(self.mid_2.size()))
            .and_then(|bytes| bytes.checked_add(self.ffn_out.size()))
            .and_then(|bytes| bytes.checked_add(self.staging.size()))
            .expect("routed-expert workspace device-byte total overflow")
    }
}

#[derive(Default)]
struct DeviceLossState {
    lost: AtomicBool,
    detail: ParkingMutex<Option<String>>,
}

impl DeviceLossState {
    fn record(&self, reason: wgpu::DeviceLostReason, message: String) {
        *self.detail.lock() = Some(format!("{reason:?}: {message}"));
        self.lost.store(true, Ordering::Release);
    }

    fn detail(&self) -> Option<String> {
        if self.lost.load(Ordering::Acquire) {
            Some(
                self.detail
                    .lock()
                    .clone()
                    .unwrap_or_else(|| "wgpu device-loss callback fired".to_string()),
            )
        } else {
            None
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum BoundedCallbackError<E> {
    DeadlineOverflow,
    Timeout,
    ChannelDisconnected,
    Callback(E),
    DeviceLost(String),
}

/// Drive a callback-based WGPU operation with nonblocking polls until it
/// completes or its deadline expires. The callback channel is inspected both
/// before and after every poll so an already-completed operation never loses
/// to a zero-duration test deadline.
fn wait_for_bounded_callback<T, E>(
    receiver: &std::sync::mpsc::Receiver<std::result::Result<T, E>>,
    timeout: Duration,
    mut poll: impl FnMut(),
    mut device_loss: impl FnMut() -> Option<String>,
) -> std::result::Result<T, BoundedCallbackError<E>> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or(BoundedCallbackError::DeadlineOverflow)?;
    loop {
        match receiver.try_recv() {
            Ok(result) => return result.map_err(BoundedCallbackError::Callback),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return Err(BoundedCallbackError::ChannelDisconnected);
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
        if let Some(detail) = device_loss() {
            return Err(BoundedCallbackError::DeviceLost(detail));
        }

        poll();

        match receiver.try_recv() {
            Ok(result) => return result.map_err(BoundedCallbackError::Callback),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return Err(BoundedCallbackError::ChannelDisconnected);
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
        if let Some(detail) = device_loss() {
            return Err(BoundedCallbackError::DeviceLost(detail));
        }
        if Instant::now() >= deadline {
            return Err(BoundedCallbackError::Timeout);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn q4_parity_readback_error<E: fmt::Debug>(
    error: BoundedCallbackError<E>,
    timeout: Duration,
) -> Q4ParityGpuError {
    match error {
        BoundedCallbackError::DeadlineOverflow => Q4ParityGpuError::new(
            Q4ParityGpuErrorKind::InvalidRequest,
            "raw Q4_0 readback deadline overflowed",
        ),
        BoundedCallbackError::Timeout => Q4ParityGpuError::new(
            Q4ParityGpuErrorKind::ReadbackTimeout,
            format!(
                "raw Q4_0 readback did not complete within {} seconds",
                timeout.as_secs_f64()
            ),
        ),
        BoundedCallbackError::ChannelDisconnected => Q4ParityGpuError::new(
            Q4ParityGpuErrorKind::ReadbackChannel,
            "raw Q4_0 readback callback channel disconnected",
        ),
        BoundedCallbackError::Callback(error) => Q4ParityGpuError::new(
            Q4ParityGpuErrorKind::ReadbackMap,
            format!("raw Q4_0 staging map failed: {error:?}"),
        ),
        BoundedCallbackError::DeviceLost(detail) => Q4ParityGpuError::new(
            Q4ParityGpuErrorKind::DeviceLost,
            format!("GPU device was lost during raw Q4_0 dispatch: {detail}"),
        ),
    }
}

struct DenseGpuResources {
    work_a: wgpu::Buffer,
    work_b: wgpu::Buffer,
    work_out: wgpu::Buffer,
    _staging_up: wgpu::Buffer,
    staging_dn: wgpu::Buffer,
    matmul_bind_group: wgpu::BindGroup,
    swiglu_bind_group: wgpu::BindGroup,
    softmax_pipeline: wgpu::ComputePipeline,
    softmax_bind_group: wgpu::BindGroup,
    conversion_scratch: ParkingMutex<Vec<f32>>,
}

struct AttentionGpuResources {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    kv_cache: GpuKvCache,
}

/// Optional component resources constructed from the authoritative GPU plan.
/// Production and hardware-independent tests share this exact factory/access
/// seam, so disabled branches cannot diverge from the methods that consume
/// the resulting resources.
struct GpuComponentResources<D, A> {
    plan: GpuResourcePlan,
    dense: Option<D>,
    attention: Option<A>,
}

impl<D, A> GpuComponentResources<D, A> {
    fn try_new(
        plan: GpuResourcePlan,
        build_dense: impl FnOnce() -> Result<D>,
        build_attention: impl FnOnce(&D) -> Result<A>,
    ) -> Result<Self> {
        let dense = if plan.constructs_dense_resources() {
            Some(build_dense()?)
        } else {
            None
        };
        let attention = if plan.constructs_attention_resources() {
            let dense = dense.as_ref().ok_or_else(|| {
                anyhow::Error::new(GpuResourceInvariantError {
                    mode: plan.mode(),
                    resource: "dense resources required by attention",
                })
            })?;
            Some(build_attention(dense)?)
        } else {
            None
        };
        Ok(Self {
            plan,
            dense,
            attention,
        })
    }

    const fn plan(&self) -> GpuResourcePlan {
        self.plan
    }

    fn require_capability(&self, capability: GpuCapability) -> Result<()> {
        self.plan
            .require_capability(capability)
            .map_err(anyhow::Error::new)
    }

    fn dense(&self, capability: GpuCapability) -> Result<&D> {
        self.require_capability(capability)?;
        self.dense.as_ref().ok_or_else(|| {
            anyhow::Error::new(GpuResourceInvariantError {
                mode: self.plan.mode(),
                resource: "dense resources",
            })
        })
    }

    fn attention(&self) -> Result<&A> {
        self.require_capability(GpuCapability::Attention)?;
        self.attention.as_ref().ok_or_else(|| {
            anyhow::Error::new(GpuResourceInvariantError {
                mode: self.plan.mode(),
                resource: "attention resources",
            })
        })
    }
}

fn validate_startup_buffer(
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
    limits: &wgpu::Limits,
) -> std::result::Result<(), GpuStartupAllocationError> {
    if size > limits.max_buffer_size {
        return Err(GpuStartupAllocationError::ExceedsMaxBufferSize {
            label: label.to_string(),
            requested: size,
            maximum: limits.max_buffer_size,
        });
    }
    let max_storage = u64::from(limits.max_storage_buffer_binding_size);
    if usage.contains(wgpu::BufferUsages::STORAGE) && size > max_storage {
        return Err(
            GpuStartupAllocationError::ExceedsMaxStorageBindingSize {
                label: label.to_string(),
                requested: size,
                maximum: max_storage,
            },
        );
    }
    Ok(())
}

fn create_startup_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> std::result::Result<wgpu::Buffer, GpuStartupAllocationError> {
    validate_startup_buffer(label, size, usage, &device.limits())?;
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    }))
}

pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// `wgpu` 0.20 has no synchronous device-loss return from submit, so
    /// dispatch checks this per-device callback state at entry and after poll.
    device_loss: Arc<DeviceLossState>,
    device_name: String,
    compute_plane: String,
    adapter_metadata: AdapterMetadata,
    expert_io: GpuExpertIoCounters,
    component_resources: GpuComponentResources<DenseGpuResources, AttentionGpuResources>,

    matmul_pipeline: Option<wgpu::ComputePipeline>,
    /// Q4_0 inline-dequant GEMV pipeline (`matmul_q4_0.wgsl`) for expert
    /// FFN passes whose weights stay in native GGUF Q4_0 blocks in VRAM.
    matmul_q4_0_pipeline: Option<wgpu::ComputePipeline>,
    swiglu_pipeline: wgpu::ComputePipeline,

    /// Serializes the *whole* dense-op execution (`matmul_into`,
    /// `swiglu_into`, `softmax`, `kv_attend`) against the backend-global
    /// `work_a`/`work_b`/`work_out`/`staging_dn` buffers and their
    /// pre-built bind groups. The `conversion_scratch` lock above only
    /// guards the host upload; without this lock two concurrent callers
    /// (the documented "two Tokio tasks share one `Arc<GpuBackend>`"
    /// case) could overwrite each other's `work_*` inputs between upload
    /// and dispatch, or double-`map_async` the single `staging_dn`
    /// readback buffer. The expert FFN path is unaffected: it runs on
    /// per-workspace buffers (`ExpertWorkspace`) and never takes this
    /// lock, so expert dispatches still overlap freely.
    dense_exec_lock: ParkingMutex<()>,

    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Value head dimension (Finding 12). Equal to `head_dim` for symmetric
    /// architectures; drives the V slice / attention-output geometry.
    v_head_dim: usize,

    /// Sole authoritative owner/index for physical routed-expert weight
    /// buffers. Identity is `(expert_id, logical_generation)`; bounded LRU
    /// lookup/install locks are released before GPU dispatch.
    physical_expert_registry: Arc<PhysicalGpuExpertRegistry<VramExpertEntry>>,

    /// Host-side logical admission policy. Every physical lookup is validated
    /// against its current admission generation before GPU execution.
    gpu_expert_cache: Arc<crate::expert_cache::GpuExpertCache>,
    /// Checked-out-on-dispatch workspaces for the expert FFN path —
    /// see [`ExpertWorkspace`]. Replaces the `expert_execution_lock`
    /// that used to serialize all expert dispatches behind one set of
    /// shared staging buffers.
    expert_workspaces: ParkingMutex<Vec<ExpertWorkspace>>,
    /// Wakes dispatchers parked on an empty workspace pool.
    expert_workspace_cv: parking_lot::Condvar,
    /// Engine-scoped truncated-payload tolerance in bytes
    /// (`RealInferencePolicy::expert_size_tolerance`), fixed at
    /// construction from config. `0` (strict, the default) requires the
    /// exact logical Q4_0 payload size on VRAM upload — never a mutable
    /// process-global switch (hardening pass, policy separation).
    q4_truncation_tolerance: usize,
}

impl GpuBackend {
    fn min_storage_buffer_offset_alignment(&self) -> u64 {
        (self
            .device
            .limits()
            .min_storage_buffer_offset_alignment as u64)
            .max(1)
    }

    fn dense_resources(
        &self,
        capability: GpuCapability,
    ) -> Result<&DenseGpuResources> {
        self.component_resources.dense(capability)
    }

    fn attention_resources(&self) -> Result<&AttentionGpuResources> {
        self.component_resources.attention()
    }

    async fn try_new(
        resource_plan: GpuResourcePlan,
        geometry: GpuBackendGeometry,
        gpu_expert_cache: Arc<crate::expert_cache::GpuExpertCache>,
    ) -> Result<Self> {
        resource_plan
            .require_capability(GpuCapability::RoutedExperts)
            .map_err(anyhow::Error::new)?;
        let GpuBackendGeometry {
            num_layers,
            max_seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            v_head_dim,
            q4_truncation_tolerance,
        } = geometry;
        // GQA models have num_kv_heads < num_heads; 0 means MHA.
        let num_kv_heads = if num_kv_heads == 0 { num_heads } else { num_kv_heads };
        // Finding 12: `v_head_dim == 0` means "symmetric" (V uses the query
        // head dim). Asymmetric-V models pass their true value head dim.
        let v_head_dim = if v_head_dim == 0 { head_dim } else { v_head_dim };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let allow_software = allow_software_wgpu_adapter();
        let mut adapters: Vec<EnumeratedAdapter> = instance
            .enumerate_adapters(wgpu::Backends::all())
            .into_iter()
            .map(|adapter| {
                let metadata = AdapterMetadata::from_info(adapter.get_info());
                EnumeratedAdapter { adapter, metadata }
            })
            .collect();

        for (index, candidate) in adapters.iter().enumerate() {
            tracing::info!(
                index,
                name = %candidate.metadata.name,
                backend = %candidate.metadata.backend,
                device_type = ?candidate.metadata.device_type,
                vendor = format_args!("{:#06x}", candidate.metadata.vendor),
                device = format_args!("{:#06x}", candidate.metadata.device),
                driver = %candidate.metadata.driver,
                driver_info = %candidate.metadata.driver_info,
                software = candidate.metadata.is_software(),
                "wgpu adapter visible"
            );
        }

        let high_performance_adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await;

        let mut high_performance_index = None;
        if let Some(adapter) = high_performance_adapter {
            let metadata = AdapterMetadata::from_info(adapter.get_info());
            high_performance_index = adapters
                .iter()
                .position(|candidate| candidate.metadata.matches(&metadata));
            if high_performance_index.is_none() {
                high_performance_index = Some(adapters.len());
                tracing::info!(
                    index = high_performance_index.unwrap(),
                    name = %metadata.name,
                    backend = %metadata.backend,
                    device_type = ?metadata.device_type,
                    vendor = format_args!("{:#06x}", metadata.vendor),
                    device = format_args!("{:#06x}", metadata.device),
                    driver = %metadata.driver,
                    driver_info = %metadata.driver_info,
                    software = metadata.is_software(),
                    "wgpu HighPerformance adapter was not returned by enumerate_adapters; adding to candidates"
                );
                adapters.push(EnumeratedAdapter { adapter, metadata });
            }
        } else {
            tracing::warn!(
                "wgpu request_adapter(HighPerformance) returned no adapter; falling back to enumerated non-CPU adapters"
            );
        }

        let metadata: Vec<AdapterMetadata> = adapters
            .iter()
            .map(|candidate| candidate.metadata.clone())
            .collect();
        let candidate_indices =
            select_wgpu_adapter_candidates(&metadata, high_performance_index, allow_software)
                .map_err(|e| match e {
                    AdapterSelectionError::NoAdapters => anyhow!(
                        "no adapters exposed by wgpu; check that the Linux Vulkan loader and a vendor ICD are installed and visible"
                    ),
                    AdapterSelectionError::OnlySoftware { count } => anyhow!(
                        "only software adapters found by wgpu ({count}); refusing to treat a software renderer as GPU. Set {ALLOW_SOFTWARE_WGPU_ADAPTER_ENV}=1 only for explicit software-adapter testing"
                    ),
                })?;

        let required_features = required_wgpu_features();
        let required_limits = required_wgpu_limits();
        let mut unsupported = Vec::new();
        let mut request_device_errors = Vec::new();
        let mut selected = None;

        for index in candidate_indices {
            let candidate = &adapters[index];
            let adapter_features = candidate.adapter.features();
            let missing_features = required_features.difference(adapter_features);
            let adapter_limits = candidate.adapter.limits();
            let mut limit_failures = Vec::new();
            required_limits.check_limits_with_fail_fn(
                &adapter_limits,
                false,
                |name, requested, available| {
                    limit_failures.push(format!(
                        "{name} required {requested}, adapter supports {available}"
                    ));
                },
            );

            if !missing_features.is_empty() || !limit_failures.is_empty() {
                tracing::warn!(
                    adapter = %candidate.metadata.summary(),
                    missing_features = ?missing_features,
                    limits = %limit_failures.join("; "),
                    "wgpu adapter rejected: required feature or limit unsupported"
                );
                unsupported.push(format!(
                    "{} missing_features={missing_features:?} limits=[{}]",
                    candidate.metadata.summary(),
                    limit_failures.join("; ")
                ));
                continue;
            }

            match candidate
                .adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("MER-GpuBackend"),
                        required_features,
                        required_limits: required_limits.clone(),
                    },
                    None,
                )
                .await
            {
                Ok((device, queue)) => {
                    selected = Some((candidate.metadata.clone(), device, queue));
                    break;
                }
                Err(e) => {
                    tracing::warn!(
                        adapter = %candidate.metadata.summary(),
                        error = %e,
                        "wgpu request_device failed"
                    );
                    request_device_errors.push(format!(
                        "{}: {e}",
                        candidate.metadata.summary()
                    ));
                }
            }
        }

        let (info, device, queue) = if let Some(selected) = selected {
            selected
        } else if !request_device_errors.is_empty() {
            return Err(anyhow!(
                "adapter found but request_device failed: {}",
                request_device_errors.join(" | ")
            ));
        } else {
            return Err(anyhow!(
                "required feature or limit unsupported by visible wgpu adapters: {}",
                unsupported.join(" | ")
            ));
        };

        let device_loss = Arc::new(DeviceLossState::default());
        let device_loss_callback = device_loss.clone();
        device.set_device_lost_callback(move |reason, message| {
            device_loss_callback.record(reason, message);
        });

        let compute_plane = wgpu_compute_plane(info.backend);
        let device_name = format!("{}-{}", compute_plane, info.name);
        tracing::info!(
            compute_plane = %compute_plane,
            adapter = %info.name,
            backend = %info.backend,
            device_type = ?info.device_type,
            vendor = format_args!("{:#06x}", info.vendor),
            device = format_args!("{:#06x}", info.device),
            driver = %info.driver,
            driver_info = %info.driver_info,
            mode = ?resource_plan.mode(),
            dense_gpu_bytes = resource_plan.dense_allocation_bytes(),
            kv_gpu_bytes = resource_plan.kv_allocation_bytes(),
            expert_workspace_gpu_bytes = resource_plan.expert_workspace_allocation_bytes(),
            "selected wgpu compute plane"
        );

        // Compile only the matmul format(s) required by the authoritative
        // component-scoped resource plan. Strict Hybrid Q4_0 therefore never
        // constructs the dense F32 shader/pipeline.
        let matmul_module = resource_plan.constructs_f32_matmul_pipeline().then(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul_shader"),
                source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
            })
        });

        let matmul_q4_0_module = resource_plan.constructs_q4_0_matmul_pipeline().then(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul_q4_0_shader"),
                source: wgpu::ShaderSource::Wgsl(MATMUL_Q4_0_SHADER.into()),
            })
        });

        let swiglu_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("swiglu_shader"),
            source: wgpu::ShaderSource::Wgsl(SWIGLU_SHADER.into()),
        });

        let softmax_module = resource_plan.constructs_dense_resources().then(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("softmax_shader"),
                source: wgpu::ShaderSource::Wgsl(SOFTMAX_SHADER.into()),
            })
        });

        // Attention/KV shader construction is absent in RoutedExpertsOnly.
        let attention_module = resource_plan.constructs_attention_resources().then(|| {
            let attention_src = ATTENTION_SHADER.replace(
                "const MAX_SEQ_LEN: u32 = 4096u;",
                &format!("const MAX_SEQ_LEN: u32 = {}u;", max_seq_len),
            ).replace(
                "const MAX_HEAD_DIM: u32 = 256u;",
                &format!("const MAX_HEAD_DIM: u32 = {}u;", head_dim),
            ).replace(
                "const MAX_V_HEAD_DIM: u32 = 256u;",
                &format!("const MAX_V_HEAD_DIM: u32 = {}u;", v_head_dim),
            );
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("attention_shader"),
                source: wgpu::ShaderSource::Wgsl(attention_src.into()),
            })
        });

        // Setup layouts manually for pipelines since push constants are used
        let layout_3_buffers = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout_3_buffers"),
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

        let layout_1_buffer = resource_plan.constructs_dense_resources().then(|| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout_1_buffer"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        }));

        let matmul_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matmul_pipeline_layout"),
            bind_group_layouts: &[&layout_3_buffers],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });

        let swiglu_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("swiglu_pipeline_layout"),
            bind_group_layouts: &[&layout_3_buffers],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });

        let softmax_pipeline_layout = layout_1_buffer.as_ref().map(|layout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("softmax_pipeline_layout"),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            })
        });

        let attention_pipeline_layout = resource_plan
            .constructs_attention_resources()
            .then(|| device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("attention_pipeline_layout"),
                bind_group_layouts: &[&layout_3_buffers],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    // 6 × u32 + 2 × u32 padding = 32 bytes, matching the
                    // 32-byte limit requested in `required_limits`.
                    range: 0..32,
                }],
            }));

        // Compute pipelines
        let matmul_pipeline = matmul_module.as_ref().map(|module| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matmul_pipeline"),
                layout: Some(&matmul_pipeline_layout),
                module,
                entry_point: "matmul_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        });

        // Same bind-group shape (read, read, read-write) and the same
        // 16-byte push-constant block as the dense pipeline, so the
        // pipeline layout is shared; only the module/entry differ.
        let matmul_q4_0_pipeline = matmul_q4_0_module.as_ref().map(|module| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matmul_q4_0_pipeline"),
                layout: Some(&matmul_pipeline_layout),
                module,
                entry_point: "matmul_q4_0_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        });

        let swiglu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("swiglu_pipeline"),
            layout: Some(&swiglu_pipeline_layout),
            module: &swiglu_module,
            entry_point: "swiglu_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let mut softmax_pipeline = softmax_module.as_ref().map(|module| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("softmax_pipeline"),
                layout: softmax_pipeline_layout.as_ref(),
                module,
                entry_point: "softmax_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        });

        let mut attention_pipeline = attention_module.as_ref().map(|module| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("attention_pipeline"),
                layout: attention_pipeline_layout.as_ref(),
                module,
                entry_point: "attention_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        });

        // Per-dispatch expert FFN workspaces — see [`ExpertWorkspace`].
        let workspace_bytes = (MAX_EXPERT_D_FF * std::mem::size_of::<f32>()) as u64;
        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let expert_workspaces: Vec<ExpertWorkspace> = (0..EXPERT_WORKSPACE_POOL)
            .map(|i| -> Result<ExpertWorkspace> {
                Ok(ExpertWorkspace {
                    x_buf: create_startup_buffer(
                        &device,
                        &format!("expert_ws{i}_x"),
                        workspace_bytes,
                        storage_usage,
                    )?,
                    mid_1: create_startup_buffer(
                        &device,
                        &format!("expert_ws{i}_mid_1"),
                        workspace_bytes,
                        storage_usage,
                    )?,
                    mid_2: create_startup_buffer(
                        &device,
                        &format!("expert_ws{i}_mid_2"),
                        workspace_bytes,
                        storage_usage,
                    )?,
                    ffn_out: create_startup_buffer(
                        &device,
                        &format!("expert_ws{i}_ffn_out"),
                        workspace_bytes,
                        storage_usage,
                    )?,
                    staging: create_startup_buffer(
                        &device,
                        &format!("expert_ws{i}_staging"),
                        workspace_bytes,
                        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    )?,
                    scratch: vec![0.0f32; MAX_EXPERT_D_FF],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let actual_expert_workspace_bytes = expert_workspaces
            .iter()
            .map(ExpertWorkspace::device_bytes)
            .try_fold(0u64, u64::checked_add)
            .ok_or_else(|| anyhow!("routed-expert workspace byte total overflow"))?;
        debug_assert_eq!(
            actual_expert_workspace_bytes,
            resource_plan.expert_workspace_allocation_bytes()
        );
        let expert_capacity_bytes = u64::try_from(gpu_expert_cache.capacity_bytes())
            .map_err(|_| anyhow!("GPU expert capacity does not fit u64"))?;
        let expert_memory_ledger = GpuMemoryLedger::new(
            actual_expert_workspace_bytes,
            expert_capacity_bytes,
        );
        let physical_expert_registry =
            PhysicalGpuExpertRegistry::new(expert_memory_ledger);

        let component_resources = GpuComponentResources::try_new(
            resource_plan,
            || {
                let storage_usage = wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC;
                let work_a =
                    create_startup_buffer(&device, "work_a", DENSE_BUFFER_BYTES, storage_usage)?;
                let work_b =
                    create_startup_buffer(&device, "work_b", DENSE_BUFFER_BYTES, storage_usage)?;
                let work_out = create_startup_buffer(
                    &device,
                    "work_out",
                    DENSE_BUFFER_BYTES,
                    storage_usage,
                )?;
                let staging_up = create_startup_buffer(
                    &device,
                    "staging_up",
                    DENSE_BUFFER_BYTES,
                    wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                )?;
                let staging_dn = create_startup_buffer(
                    &device,
                    "staging_dn",
                    DENSE_BUFFER_BYTES,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                )?;
                let matmul_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("matmul_bind_group"),
                    layout: &layout_3_buffers,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: work_a.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: work_b.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: work_out.as_entire_binding(),
                        },
                    ],
                });
                let swiglu_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("swiglu_bind_group"),
                    layout: &layout_3_buffers,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: work_a.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: work_b.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: work_out.as_entire_binding(),
                        },
                    ],
                });
                let softmax_layout = layout_1_buffer.as_ref().ok_or_else(|| {
                    anyhow::Error::new(GpuResourceInvariantError {
                        mode: resource_plan.mode(),
                        resource: "softmax bind-group layout",
                    })
                })?;
                let softmax_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("softmax_bind_group"),
                    layout: softmax_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: work_a.as_entire_binding(),
                    }],
                });
                let softmax_pipeline = softmax_pipeline.take().ok_or_else(|| {
                    anyhow::Error::new(GpuResourceInvariantError {
                        mode: resource_plan.mode(),
                        resource: "softmax pipeline",
                    })
                })?;
                Ok(DenseGpuResources {
                    work_a,
                    work_b,
                    work_out,
                    _staging_up: staging_up,
                    staging_dn,
                    matmul_bind_group,
                    swiglu_bind_group,
                    softmax_pipeline,
                    softmax_bind_group,
                    conversion_scratch: ParkingMutex::new(vec![0.0f32; DENSE_WORK_MAX_ELEMS]),
                })
            },
            |dense| {
                let k_dim = num_kv_heads.checked_mul(head_dim).ok_or_else(|| {
                    anyhow!("GPU KV key geometry overflow after validation")
                })?;
                let v_dim = num_kv_heads.checked_mul(v_head_dim).ok_or_else(|| {
                    anyhow!("GPU KV value geometry overflow after validation")
                })?;
                let kv_cache_buffer = create_startup_buffer(
                    &device,
                    "kv_cache",
                    resource_plan.kv_allocation_bytes(),
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                )?;
                let kv_cache = GpuKvCache {
                    buffer: kv_cache_buffer,
                    num_layers,
                    max_seq_len,
                    k_dim,
                    v_dim,
                };
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("attention_bind_group"),
                    layout: &layout_3_buffers,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: dense.work_a.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: kv_cache.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dense.work_out.as_entire_binding(),
                        },
                    ],
                });
                let pipeline = attention_pipeline.take().ok_or_else(|| {
                    anyhow::Error::new(GpuResourceInvariantError {
                        mode: resource_plan.mode(),
                        resource: "attention pipeline",
                    })
                })?;
                Ok(AttentionGpuResources {
                    pipeline,
                    bind_group,
                    kv_cache,
                })
            },
        )?;

        Ok(Self {
            device,
            queue,
            device_loss,
            device_name,
            compute_plane,
            adapter_metadata: info,
            expert_io: GpuExpertIoCounters::default(),
            component_resources,
            matmul_pipeline,
            matmul_q4_0_pipeline,
            swiglu_pipeline,
            dense_exec_lock: ParkingMutex::new(()),
            num_heads,
            num_kv_heads,
            head_dim,
            v_head_dim,
            physical_expert_registry,
            gpu_expert_cache,
            expert_workspaces: ParkingMutex::new(expert_workspaces),
            expert_workspace_cv: parking_lot::Condvar::new(),
            q4_truncation_tolerance,
        })
    }

    /// Validate every PR1/PR2 upload invariant and compute the exact wgpu
    /// buffer descriptor bytes before the physical registry reserves space.
    fn plan_expert_upload(
        &self,
        dtype: crate::inference::WeightDtype,
        weight_bytes: &[u8],
        d_model: usize,
        d_ff: usize,
    ) -> anyhow::Result<ExpertUploadPlan> {
        match dtype {
            crate::inference::WeightDtype::Q4_0 => self
                .plan_expert_upload_q4_0(weight_bytes, d_model, d_ff)
                .map(ExpertUploadPlan::Q4_0),
            _ => self
                .plan_expert_upload_f32(weight_bytes, d_model, d_ff)
                .map(ExpertUploadPlan::F32),
        }
    }

    fn plan_expert_upload_f32(
        &self,
        weight_bytes: &[u8],
        d_model: usize,
        d_ff: usize,
    ) -> anyhow::Result<F32ExpertUploadPlan> {
        anyhow::ensure!(
            d_model > 0 && d_ff > 0,
            "invalid expert shape: d_ff={} d_model={} produces zero-byte projections",
            d_ff,
            d_model
        );
        let spec = RoutedExpertGpuSpec {
            dtype: crate::inference::WeightDtype::F32,
            d_model,
            d_ff,
        };
        // Re-run startup compatibility at upload as a defensive check. Both
        // paths share the checked projection-layout formula below.
        routed_expert_gpu_compatibility(spec).map_err(|e| anyhow!(e))?;
        let layout = f32_expert_projection_layout(
            spec,
            self.min_storage_buffer_offset_alignment(),
        )
        .map_err(|e| anyhow!(e))?;
        f32_expert_upload_plan(layout, weight_bytes.len(), d_model, d_ff)
    }

    fn plan_expert_upload_q4_0(
        &self,
        weight_bytes: &[u8],
        d_model: usize,
        d_ff: usize,
    ) -> anyhow::Result<Q4ExpertUploadPlan> {
        use crate::inference::{Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS};

        anyhow::ensure!(
            d_model > 0 && d_ff > 0,
            "invalid expert shape: d_ff={} d_model={}",
            d_ff, d_model
        );
        routed_expert_gpu_compatibility(RoutedExpertGpuSpec {
            dtype: crate::inference::WeightDtype::Q4_0,
            d_model,
            d_ff,
        })
        .map_err(|e| anyhow!(e))?;
        let proj_elems = d_ff
            .checked_mul(d_model)
            .ok_or_else(|| anyhow!("Q4_0 expert shape overflow: d_ff={d_ff} d_model={d_model}"))?;
        let blocks_per_projection = proj_elems / Q4_0_BLOCK_ELEMS;
        let projection_bytes = blocks_per_projection
            .checked_mul(Q4_0_BLOCK_BYTES)
            .ok_or_else(|| anyhow!(
                "Q4_0 expert projection byte size overflow: {blocks_per_projection} blocks × {Q4_0_BLOCK_BYTES}B"
            ))?;
        let required_bytes = projection_bytes
            .checked_mul(3)
            .ok_or_else(|| anyhow!(
                "Q4_0 expert total byte size overflow: 3 × {projection_bytes}B"
            ))?;
        let tolerance = self.q4_truncation_tolerance;
        anyhow::ensure!(
            weight_bytes.len() >= required_bytes
                || (required_bytes > tolerance
                    && tolerance > 0
                    && required_bytes - weight_bytes.len() <= tolerance),
            "Q4_0 expert weight buffer too small: got {} bytes, need {} (3 × {} blocks × {}B); \
             missing logical bytes are never zero-filled in strict mode \
             (allow_truncated_expert_payloads = false)",
            weight_bytes.len(),
            required_bytes,
            blocks_per_projection,
            Q4_0_BLOCK_BYTES
        );
        let down_block_offset = blocks_per_projection
            .checked_mul(2)
            .ok_or_else(|| anyhow!("Q4_0 expert block offset overflow"))?;
        anyhow::ensure!(
            down_block_offset <= u32::MAX as usize,
            "Q4_0 expert block count {} exceeds u32 push-constant range",
            down_block_offset
        );
        let padded_bytes = required_bytes
            .checked_add(3)
            .ok_or_else(|| anyhow!("Q4_0 padded device byte size overflow"))?
            / 4
            * 4;
        Ok(Q4ExpertUploadPlan {
            device_bytes: u64::try_from(padded_bytes)
                .map_err(|_| anyhow!("Q4_0 padded buffer length does not fit u64"))?,
            required_bytes,
            up_block_offset: blocks_per_projection as u32,
            down_block_offset: down_block_offset as u32,
        })
    }

    /// Upload validated F32 bytes after capacity has been reserved. Weight
    /// layout is `[gate | up | down]`; `projection_offset` was checked against
    /// the selected device's storage-offset alignment by the upload plan.
    fn build_expert_entry(
        &self,
        key: PhysicalExpertKey,
        weight_bytes: &[u8],
        d_model: usize,
        d_ff: usize,
        plan: F32ExpertUploadPlan,
        permit: &mut PhysicalInstallPermit<VramExpertEntry>,
    ) -> anyhow::Result<VramExpertEntry> {

        // ── Upload weights to VRAM ────────────────────────────────────────────
        let weight_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("vram_expert_weights"),
            size:               plan.device_bytes,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        debug_assert_eq!(weight_buf.size(), plan.device_bytes);
        let upload_bytes = usize::try_from(plan.device_bytes)
            .map_err(|_| anyhow!("F32 device buffer size does not fit usize"))?;
        self.queue
            .write_buffer(&weight_buf, 0, &weight_bytes[..upload_bytes]);
        self.expert_io
            .expert_weight_uploads
            .fetch_add(1, Ordering::Relaxed);
        self.expert_io
            .expert_weight_upload_bytes
            .fetch_add(plan.device_bytes, Ordering::Relaxed);
        let allocation = permit.charge_allocation().map_err(|e| anyhow!(e))?;

        Ok(VramExpertEntry {
            key,
            weight_buf,
            d_model,
            d_ff,
            layout: VramWeightLayout::F32,
            proj_bytes: plan.projection_offset,
            up_block_off: 0,
            down_block_off: 0,
            device_bytes: plan.device_bytes,
            _allocation: allocation,
        })
    }

    /// Upload a **native Q4_0** expert weight buffer to VRAM for the
    /// inline-dequant pipeline (`matmul_q4_0.wgsl`). Unlike
    /// [`Self::build_expert_entry`] the bytes
    /// are *not* dequantised first: the GGUF Q4_0 blocks (18 bytes per 32
    /// weights) cross PCIe and live in VRAM as-is — ~8× fewer bytes than
    /// the dense F32 stream — and each compute pass unpacks blocks inline.
    ///
    /// Expected layout (matching `OwnedExpertWeights::from_bytes_q4_0`):
    /// gate, up and down block streams concatenated back-to-back, each
    /// `(d_ff·d_model / 32) × 18` bytes. Both `d_model` and `d_ff` must be
    /// multiples of the 32-element Q4_0 block (the caller guarantees this
    /// via `routed_expert_gpu_compatibility`), so every matrix row starts on
    /// a block boundary. Buffers short by at most one page are zero-padded,
    /// mirroring the CPU loader's `q4_expert_bytes_with_tolerance`.
    fn build_expert_entry_q4_0(
        &self,
        key: PhysicalExpertKey,
        weight_bytes: &[u8],
        d_model: usize,
        d_ff: usize,
        plan: Q4ExpertUploadPlan,
        permit: &mut PhysicalInstallPermit<VramExpertEntry>,
    ) -> anyhow::Result<VramExpertEntry> {
        // wgpu requires buffer sizes / write lengths to be 4-byte
        // multiples; the logical payload is only guaranteed even, so
        // round up and zero-fill the tail (also covers the ≤ one-page
        // shortfall tolerance above).
        let padded_len = usize::try_from(plan.device_bytes)
            .map_err(|_| anyhow!("Q4_0 device buffer size does not fit usize"))?;
        let weight_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("vram_expert_weights_q4_0"),
            size:               plan.device_bytes,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        debug_assert_eq!(weight_buf.size(), plan.device_bytes);
        let avail = weight_bytes.len().min(plan.required_bytes);
        if avail == padded_len {
            // Fast path: the source covers the full (already 4-byte-
            // aligned) buffer, so write it directly without a copy.
            self.queue.write_buffer(&weight_buf, 0, &weight_bytes[..padded_len]);
        } else {
            // Source is short of `padded_len` — either `need` itself
            // isn't a 4-byte multiple, or the buffer is within the
            // one-page shortfall tolerance. Zero-fill the tail.
            let mut padded = Vec::with_capacity(padded_len);
            padded.extend_from_slice(&weight_bytes[..avail]);
            padded.resize(padded_len, 0);
            self.queue.write_buffer(&weight_buf, 0, &padded);
        }
        self.expert_io
            .expert_weight_uploads
            .fetch_add(1, Ordering::Relaxed);
        self.expert_io
            .expert_weight_upload_bytes
            .fetch_add(plan.device_bytes, Ordering::Relaxed);
        let allocation = permit.charge_allocation().map_err(|e| anyhow!(e))?;

        // The projection base is selected via the `w_block_off` push
        // constant (Q4_0 blocks are 18 bytes and cannot honour
        // min_storage_buffer_offset_alignment), so the per-dispatch
        // matmul bind groups bind the entire weight buffer.
        Ok(VramExpertEntry {
            key,
            weight_buf,
            d_model,
            d_ff,
            layout: VramWeightLayout::Q4_0,
            proj_bytes: 0,
            up_block_off: plan.up_block_offset,
            down_block_off: plan.down_block_offset,
            device_bytes: plan.device_bytes,
            _allocation: allocation,
        })
    }

    /// Check an [`ExpertWorkspace`] out of the pool, parking on the
    /// condvar until one frees up when all are in flight. With
    /// [`EXPERT_WORKSPACE_POOL`] workspaces, up to that many expert
    /// FFN dispatches proceed concurrently; the (rare) wait here
    /// replaces the old whole-path `expert_execution_lock`.
    ///
    /// Fairness: `parking_lot::Condvar` wakes waiters FIFO-ish but
    /// makes no strict guarantee; with a pool of
    /// [`EXPERT_WORKSPACE_POOL`] = 4 against a typical MoE top-K of
    /// 2–4 concurrent dispatches, contention (let alone starvation)
    /// is not expected in practice.
    fn acquire_expert_workspace(&self) -> ExpertWorkspace {
        let mut pool = self.expert_workspaces.lock();
        loop {
            if let Some(ws) = pool.pop() {
                return ws;
            }
            self.expert_workspace_cv.wait(&mut pool);
        }
    }

    fn release_expert_workspace(&self, ws: ExpertWorkspace) {
        self.expert_workspaces.lock().push(ws);
        self.expert_workspace_cv.notify_one();
    }

    /// Dispatch a SwiGLU expert FFN where the weight buffer is already
    /// VRAM-resident. Uploads only `x` (hidden state, ~8 KB); the weights
    /// never cross PCIe.
    ///
    /// The weight layout assumed is `[gate_proj || up_proj || down_proj]`
    /// matching `ExpertWeights::from_bytes` / the SwiGLU forward convention.
    ///
    /// **Concurrency / async pipeline.** Each call checks a private
    /// [`ExpertWorkspace`] out of the pool, encodes against that
    /// workspace's buffers, and waits only for **its own** submission
    /// (`Maintain::wait_for(submission_index)`) — not for the whole
    /// device queue (`Maintain::Wait`) the way the legacy path did.
    /// Concurrent expert dispatches therefore overlap on the queue:
    /// while one dispatch is in its readback, another can upload,
    /// encode and submit.
    fn expert_matmul_from_vram(
        &self,
        layer:  u32,
        expert_id: u32,
        entry: &VramExpertEntry,
        x:     TensorView<'_>,
        out:   &mut TensorViewMut<'_>,
    ) -> std::result::Result<(), GpuExpertDispatchError> {
        let mut ws = self.acquire_expert_workspace();
        let result = self.expert_ffn_dispatch(layer, expert_id, entry, x, out, &mut ws);
        // Always return the workspace — including on error paths — or
        // the pool would leak a slot per failed dispatch.
        self.release_expert_workspace(ws);
        result
    }

    fn expert_ffn_dispatch(
        &self,
        layer:  u32,
        expert_id: u32,
        entry: &VramExpertEntry,
        x:     TensorView<'_>,
        out:   &mut TensorViewMut<'_>,
        ws:    &mut ExpertWorkspace,
    ) -> std::result::Result<(), GpuExpertDispatchError> {
        use std::num::NonZeroU64;

        let d_model = entry.d_model;
        let d_ff    = entry.d_ff;

        if let Some(detail) = self.device_loss.detail() {
            return Err(GpuExpertDispatchError::new(
                layer, expert_id, GpuExpertDispatchErrorKind::DeviceLost, detail,
            ));
        }
        if x.data.len() != d_model || out.data.len() != d_model {
            return Err(GpuExpertDispatchError::new(
                layer,
                expert_id,
                GpuExpertDispatchErrorKind::RuntimeInvariant,
                format!(
                    "dispatch buffers disagree with resident geometry: x={} out={} d_model={d_model}",
                    x.data.len(), out.data.len()
                ),
            ));
        }

        // ── Upload x to the workspace's private x_buf ─────────────────────────
        // Per-workspace scratch: no contention with other dispatches or
        // with the dense ops' backend-global conversion scratch.
        if d_model > ws.scratch.len() {
            return Err(GpuExpertDispatchError::new(
                layer,
                expert_id,
                GpuExpertDispatchErrorKind::RuntimeInvariant,
                format!("d_model {d_model} exceeds GPU expert workspace {}", ws.scratch.len()),
            ));
        }
        for i in 0..d_model {
            ws.scratch[i] = x.data[i].to_f32();
        }
        self.queue.write_buffer(&ws.x_buf, 0, bytemuck::cast_slice(&ws.scratch[..d_model]));
        let hidden_bytes = u64::try_from(d_model * std::mem::size_of::<f32>())
            .expect("routed-expert hidden upload length fits u64");
        self.expert_io
            .hidden_state_uploads
            .fetch_add(1, Ordering::Relaxed);
        self.expert_io
            .hidden_state_upload_bytes
            .fetch_add(hidden_bytes, Ordering::Relaxed);

        // ── Per-dispatch bind groups against the workspace buffers ────────────
        // Bind group creation is microseconds against a millisecond-scale
        // submit+readback; paying it per dispatch is what frees the expert
        // path from the shared `work_*` buffers (and the execution lock
        // that serialized them).
        let matmul_pipeline = match entry.layout {
            VramWeightLayout::F32 => self.matmul_pipeline.as_ref(),
            VramWeightLayout::Q4_0 => self.matmul_q4_0_pipeline.as_ref(),
        }
        .ok_or_else(|| {
            GpuExpertDispatchError::new(
                layer,
                expert_id,
                GpuExpertDispatchErrorKind::RuntimeInvariant,
                format!(
                    "routed-expert {:?} pipeline is unavailable in {:?} backend mode",
                    entry.layout,
                    self.component_resources.plan().mode()
                ),
            )
        })?;
        let matmul_bgl = matmul_pipeline.get_bind_group_layout(0);
        let swiglu_bgl = self.swiglu_pipeline.get_bind_group_layout(0);

        // Weight binding for a projection pass. F32 binds the projection's
        // sub-range (offsets validated in `build_expert_entry`); Q4_0 binds
        // the whole buffer and selects the base via `w_block_off`.
        let weight_binding = |proj: u32| -> wgpu::BindingResource<'_> {
            match entry.layout {
                VramWeightLayout::F32 => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &entry.weight_buf,
                    offset: proj as u64 * entry.proj_bytes,
                    size:   NonZeroU64::new(entry.proj_bytes),
                }),
                VramWeightLayout::Q4_0 => entry.weight_buf.as_entire_binding(),
            }
        };
        let make_matmul_bg = |label: &str, proj: u32, x_buf: &wgpu::Buffer, out_buf: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some(label),
                layout:  &matmul_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: weight_binding(proj) },
                    wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                ],
            })
        };

        // Pass 1: gate matmul — weight[gate] × x → mid_1.
        let gate_bg = make_matmul_bg("expert_gate_bg", 0, &ws.x_buf, &ws.mid_1);
        // Pass 2: up matmul — weight[up] × x → mid_2.
        let up_bg = make_matmul_bg("expert_up_bg", 1, &ws.x_buf, &ws.mid_2);
        // Pass 3: SwiGLU — mid_1, mid_2 → ffn_out.
        let swiglu_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("expert_swiglu_bg"),
            layout:  &swiglu_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ws.mid_1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ws.mid_2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ws.ffn_out.as_entire_binding() },
            ],
        });
        // Pass 4: down matmul — weight[down] × ffn_out → mid_1.
        let down_bg = make_matmul_bg("expert_down_bg", 2, &ws.ffn_out, &ws.mid_1);

        // ── Single command buffer: 4 sequential compute passes ───────────────
        // The GPU executes these in order; no host-side synchronization needed
        // between passes. One submit = one PCIe round-trip for the whole FFN.
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("expert_ffn_encoder"),
        });

        // Pass 1: gate_proj × x → mid_1   (M=d_ff, K=d_model, N=1)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("expert_gate_pass"),
                timestamp_writes: None,
            });
            match entry.layout {
                VramWeightLayout::F32 => {
                    cpass.set_pipeline(matmul_pipeline);
                    cpass.set_bind_group(0, &gate_bg, &[]);
                    cpass.set_push_constants(0, bytemuck::bytes_of(&MatmulPushConstants {
                        m: d_ff as u32, n: 1, k: d_model as u32, w_block_off: 0,
                    }));
                    cpass.dispatch_workgroups(1, (d_ff as u32 + 15) / 16, 1);
                }
                VramWeightLayout::Q4_0 => {
                    cpass.set_pipeline(matmul_pipeline);
                    cpass.set_bind_group(0, &gate_bg, &[]);
                    cpass.set_push_constants(0, bytemuck::bytes_of(&MatmulPushConstants {
                        m: d_ff as u32, n: 1, k: d_model as u32, w_block_off: 0,
                    }));
                    cpass.dispatch_workgroups((d_ff as u32 + 63) / 64, 1, 1);
                }
            }
        }

        // Pass 2: up_proj × x → mid_2   (M=d_ff, K=d_model, N=1)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("expert_up_pass"),
                timestamp_writes: None,
            });
            match entry.layout {
                VramWeightLayout::F32 => {
                    cpass.set_pipeline(matmul_pipeline);
                    cpass.set_bind_group(0, &up_bg, &[]);
                    cpass.set_push_constants(0, bytemuck::bytes_of(&MatmulPushConstants {
                        m: d_ff as u32, n: 1, k: d_model as u32, w_block_off: 0,
                    }));
                    cpass.dispatch_workgroups(1, (d_ff as u32 + 15) / 16, 1);
                }
                VramWeightLayout::Q4_0 => {
                    cpass.set_pipeline(matmul_pipeline);
                    cpass.set_bind_group(0, &up_bg, &[]);
                    cpass.set_push_constants(0, bytemuck::bytes_of(&MatmulPushConstants {
                        m: d_ff as u32, n: 1, k: d_model as u32,
                        w_block_off: entry.up_block_off,
                    }));
                    cpass.dispatch_workgroups((d_ff as u32 + 63) / 64, 1, 1);
                }
            }
        }

        // Pass 3: SwiGLU(mid_1, mid_2) → ffn_out   (n_elements=d_ff)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("expert_swiglu_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.swiglu_pipeline);
            cpass.set_bind_group(0, &swiglu_bg, &[]);
            cpass.set_push_constants(0, bytemuck::bytes_of(&SwigluPushConstants {
                n_elements: d_ff as u32,
                swiglu_limit: crate::inference::swiglu_limit().unwrap_or(f32::INFINITY),
                _pad1: 0, _pad2: 0,
            }));
            let wg_x = (d_ff as u32 + 255) / 256;
            cpass.dispatch_workgroups(wg_x, 1, 1);
        }

        // Pass 4: down_proj × ffn_out → mid_1   (M=d_model, K=d_ff, N=1)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("expert_down_pass"),
                timestamp_writes: None,
            });
            match entry.layout {
                VramWeightLayout::F32 => {
                    cpass.set_pipeline(matmul_pipeline);
                    cpass.set_bind_group(0, &down_bg, &[]);
                    cpass.set_push_constants(0, bytemuck::bytes_of(&MatmulPushConstants {
                        m: d_model as u32, n: 1, k: d_ff as u32, w_block_off: 0,
                    }));
                    cpass.dispatch_workgroups(1, (d_model as u32 + 15) / 16, 1);
                }
                VramWeightLayout::Q4_0 => {
                    cpass.set_pipeline(matmul_pipeline);
                    cpass.set_bind_group(0, &down_bg, &[]);
                    cpass.set_push_constants(0, bytemuck::bytes_of(&MatmulPushConstants {
                        m: d_model as u32, n: 1, k: d_ff as u32,
                        w_block_off: entry.down_block_off,
                    }));
                    cpass.dispatch_workgroups((d_model as u32 + 63) / 64, 1, 1);
                }
            }
        }

        // ── Readback mid_1 → out ──────────────────────────────────────────────
        let out_bytes = (d_model * 4) as u64;
        encoder.copy_buffer_to_buffer(&ws.mid_1, 0, &ws.staging, 0, out_bytes);
        // Wait only for *this* submission — other in-flight expert
        // dispatches (and dense ops) keep making progress on the queue.
        let submission = self.queue.submit(Some(encoder.finish()));
        self.expert_io
            .queue_submissions
            .fetch_add(1, Ordering::Relaxed);

        let slice = ws.staging.slice(0..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        self.expert_io.map_requests.fetch_add(1, Ordering::Relaxed);
        slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
        self.device.poll(wgpu::Maintain::wait_for(submission));

        if let Some(detail) = self.device_loss.detail() {
            return Err(GpuExpertDispatchError::new(
                layer, expert_id, GpuExpertDispatchErrorKind::DeviceLost, detail,
            ));
        }

        rx.recv()
            .map_err(|e| GpuExpertDispatchError::new(
                layer,
                expert_id,
                GpuExpertDispatchErrorKind::ReadbackChannel,
                format!("expert readback callback channel failed: {e:?}"),
            ))?
            .map_err(|e| GpuExpertDispatchError::new(
                layer,
                expert_id,
                GpuExpertDispatchErrorKind::ReadbackMap,
                format!("expert readback buffer map failed: {e:?}"),
            ))?;

        {
            use half::slice::HalfFloatSliceExt;
            let view   = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&view);
            // Vectorized f32 → f16 downcast. `half`'s slice conversion does
            // runtime CPU-feature detection (F16C/AVX2/AVX-512), so this picks
            // up hardware float-to-half on capable hosts without compile-time
            // target-feature gating, and falls back to scalar elsewhere.
            out.data[..d_model].convert_from_f32_slice(&floats[..d_model]);
        }
        ws.staging.unmap();
        self.expert_io
            .readback_completions
            .fetch_add(1, Ordering::Relaxed);
        self.expert_io
            .readback_bytes
            .fetch_add(out_bytes, Ordering::Relaxed);
        Ok(())
    }
}

impl Backend for GpuBackend {
    fn device_name(&self) -> &str {
        &self.device_name
    }

    fn is_gpu(&self) -> bool {
        true
    }

    fn matmul_into(&self, a: TensorView, b: TensorView, out: &mut TensorViewMut) -> Result<()> {
        let dense = self.dense_resources(GpuCapability::Dense)?;
        let matmul_pipeline = self.matmul_pipeline.as_ref().ok_or_else(|| {
            anyhow!("full GPU resource plan is missing the dense F32 matmul pipeline")
        })?;
        // Serialize the whole op: the shared `work_*`/`staging_dn` buffers
        // and bind groups can't be safely shared across concurrent callers
        // (see `dense_exec_lock`). Held until readback completes.
        let _exec = self.dense_exec_lock.lock();
        let a_len = a.data.len();
        let b_len = b.data.len();
        let out_len = out.rows * out.cols;

        // Host-side conversions + uploads. The `dense_exec_lock` already
        // serializes callers, so `conversion_scratch` is uncontended here.
        {
            let mut scratch = dense.conversion_scratch.lock();
            assert!(a_len <= scratch.len());
            assert!(b_len <= scratch.len());
            assert!(out_len <= scratch.len());

            // Upload A
            for i in 0..a_len {
                scratch[i] = a.data[i].to_f32();
            }
            self.queue.write_buffer(&dense.work_a, 0, bytemuck::cast_slice(&scratch[..a_len]));

            // Upload B
            for i in 0..b_len {
                scratch[i] = b.data[i].to_f32();
            }
            self.queue.write_buffer(&dense.work_b, 0, bytemuck::cast_slice(&scratch[..b_len]));
        }

        // Dispatch
        let pcs = MatmulPushConstants {
            m: a.rows as u32,
            n: b.cols as u32,
            k: a.cols as u32,
            w_block_off: 0,
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(matmul_pipeline);
            compute_pass.set_bind_group(0, &dense.matmul_bind_group, &[]);
            compute_pass.set_push_constants(0, bytemuck::bytes_of(&pcs));
            compute_pass.dispatch_workgroups(
                (b.cols as u32 + 15) / 16,
                (a.rows as u32 + 15) / 16,
                1,
            );
        }

        // Readback
        let out_bytes = (out_len * 4) as u64;
        encoder.copy_buffer_to_buffer(&dense.work_out, 0, &dense.staging_dn, 0, out_bytes);
        // Wait only for this submission, not the whole device queue.
        let submission = self.queue.submit(Some(encoder.finish()));

        let slice = dense.staging_dn.slice(0..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::wait_for(submission));

        rx.recv()
            .map_err(|e| anyhow!("Channel error on GPU readback: {:?}", e))?
            .map_err(|e| anyhow!("Buffer map error on GPU readback: {:?}", e))?;

        {
            let view = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&view);
            for i in 0..out_len {
                out.data[i] = half::f16::from_f32(floats[i]);
            }
        }
        dense.staging_dn.unmap();
        Ok(())
    }

    fn swiglu_into(&self, gate: TensorView, up: TensorView, out: &mut TensorViewMut) -> Result<()> {
        let dense = self.dense_resources(GpuCapability::Dense)?;
        // Serialize the whole op against the shared `work_*`/`staging_dn`
        // buffers (see `dense_exec_lock`).
        let _exec = self.dense_exec_lock.lock();
        let len = gate.data.len();
        let out_len = out.rows * out.cols;
        assert_eq!(up.data.len(), len);
        assert_eq!(out_len, len);

        // Host-side conversions + uploads (serialized by `dense_exec_lock`).
        {
            let mut scratch = dense.conversion_scratch.lock();
            assert!(len <= scratch.len());

            // Upload gate
            for i in 0..len {
                scratch[i] = gate.data[i].to_f32();
            }
            self.queue.write_buffer(&dense.work_a, 0, bytemuck::cast_slice(&scratch[..len]));

            // Upload up
            for i in 0..len {
                scratch[i] = up.data[i].to_f32();
            }
            self.queue.write_buffer(&dense.work_b, 0, bytemuck::cast_slice(&scratch[..len]));
        }

        // Dispatch
        let pcs = SwigluPushConstants {
            n_elements: len as u32,
            swiglu_limit: crate::inference::swiglu_limit().unwrap_or(f32::INFINITY),
            _pad1: 0,
            _pad2: 0,
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("swiglu_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("swiglu_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.swiglu_pipeline);
            compute_pass.set_bind_group(0, &dense.swiglu_bind_group, &[]);
            compute_pass.set_push_constants(0, bytemuck::bytes_of(&pcs));
            compute_pass.dispatch_workgroups((len as u32 + 255) / 256, 1, 1);
        }

        // Readback
        let out_bytes = (len * 4) as u64;
        encoder.copy_buffer_to_buffer(&dense.work_out, 0, &dense.staging_dn, 0, out_bytes);
        // Wait only for this submission, not the whole device queue.
        let submission = self.queue.submit(Some(encoder.finish()));

        let slice = dense.staging_dn.slice(0..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::wait_for(submission));

        rx.recv()
            .map_err(|e| anyhow!("Channel error on GPU readback: {:?}", e))?
            .map_err(|e| anyhow!("Buffer map error on GPU readback: {:?}", e))?;

        {
            let view = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&view);
            for i in 0..len {
                out.data[i] = half::f16::from_f32(floats[i]);
            }
        }
        dense.staging_dn.unmap();
        Ok(())
    }

    fn softmax(&self, x: &mut TensorViewMut) -> Result<()> {
        let dense = self.dense_resources(GpuCapability::Dense)?;
        // Serialize the whole op against the shared `work_a`/`staging_dn`
        // buffers (see `dense_exec_lock`).
        let _exec = self.dense_exec_lock.lock();
        let len = x.data.len();

        // Host-side upload (serialized by `dense_exec_lock`).
        {
            let mut scratch = dense.conversion_scratch.lock();
            assert!(len <= scratch.len());

            // Upload x
            for i in 0..len {
                scratch[i] = x.data[i].to_f32();
            }
            self.queue.write_buffer(&dense.work_a, 0, bytemuck::cast_slice(&scratch[..len]));
        }

        // Dispatch
        let pcs = SoftmaxPushConstants {
            rows: x.rows as u32,
            cols: x.cols as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("softmax_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&dense.softmax_pipeline);
            compute_pass.set_bind_group(0, &dense.softmax_bind_group, &[]);
            compute_pass.set_push_constants(0, bytemuck::bytes_of(&pcs));
            compute_pass.dispatch_workgroups(x.rows as u32, 1, 1);
        }

        // Readback from work_a (in-place)
        let out_bytes = (len * 4) as u64;
        encoder.copy_buffer_to_buffer(&dense.work_a, 0, &dense.staging_dn, 0, out_bytes);
        // Wait only for this submission, not the whole device queue.
        let submission = self.queue.submit(Some(encoder.finish()));

        let slice = dense.staging_dn.slice(0..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::wait_for(submission));

        rx.recv()
            .map_err(|e| anyhow!("Channel error on GPU readback: {:?}", e))?
            .map_err(|e| anyhow!("Buffer map error on GPU readback: {:?}", e))?;

        {
            let view = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&view);
            for i in 0..len {
                x.data[i] = half::f16::from_f32(floats[i]);
            }
        }
        dense.staging_dn.unmap();
        Ok(())
    }

    fn kv_cache_insert(
        &self,
        _layer: usize,
        _position: usize,
        _k: TensorView,
        _v: TensorView,
    ) -> Result<()> {
        self.component_resources
            .require_capability(GpuCapability::Kv)?;
        // The VRAM KV cache is process-wide and addressed only by
        // `(layer, position)`. BatchScheduler can run multiple requests at
        // the same position concurrently against this backend, so using the
        // GPU KV path would let those requests overwrite each other's slots.
        // Fail safely and let transformer.rs use its per-request CPU KvCache
        // until the GPU cache grows a request/session namespace.
        anyhow::bail!(
            "GPU KV cache is disabled because it is not request-isolated under concurrent batching"
        )
    }

    fn kv_attend(
        &self,
        layer: usize,
        q: TensorView,
        seq_len: usize,
        out: &mut TensorViewMut,
    ) -> Result<()> {
        let dense = self.dense_resources(GpuCapability::Attention)?;
        let attention = self.attention_resources()?;
        // Serialize the whole op against the shared `work_*`/`staging_dn`
        // buffers (see `dense_exec_lock`).
        let _exec = self.dense_exec_lock.lock();
        let q_len = q.data.len();
        let out_len = out.rows * out.cols;

        // Host-side Q upload (serialized by `dense_exec_lock`).
        {
            let mut scratch = dense.conversion_scratch.lock();
            assert!(q_len <= scratch.len());
            assert!(out_len <= scratch.len());

            // Upload Q
            for i in 0..q_len {
                scratch[i] = q.data[i].to_f32();
            }
            self.queue.write_buffer(&dense.work_a, 0, bytemuck::cast_slice(&scratch[..q_len]));
        }

        // Dispatch
        // Pass the layer offset in f32 *elements*: a byte offset cast to
        // u32 silently wraps past 4 GiB for deep models with large KV
        // slices. Guard the (4× larger) element range explicitly.
        let layer_off_elems = attention.kv_cache.offset_bytes(layer, 0, 0) / 4;
        if layer_off_elems > u32::MAX as u64 {
            return Err(anyhow!(
                "KV layer offset {layer_off_elems} elements exceeds u32 push-constant range"
            ));
        }
        let pcs = AttentionPushConstants {
            num_heads: self.num_heads as u32,
            num_kv_heads: self.num_kv_heads as u32,
            head_dim: self.head_dim as u32,
            seq_len: seq_len as u32,
            layer_offset: layer_off_elems as u32,
            v_head_dim: self.v_head_dim as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attention_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attention_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&attention.pipeline);
            compute_pass.set_bind_group(0, &attention.bind_group, &[]);
            compute_pass.set_push_constants(0, bytemuck::bytes_of(&pcs));
            compute_pass.dispatch_workgroups(self.num_heads as u32, 1, 1);
        }

        // Readback
        let out_bytes = (out_len * 4) as u64;
        encoder.copy_buffer_to_buffer(&dense.work_out, 0, &dense.staging_dn, 0, out_bytes);
        // Wait only for this submission, not the whole device queue.
        let submission = self.queue.submit(Some(encoder.finish()));

        let slice = dense.staging_dn.slice(0..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::wait_for(submission));

        rx.recv()
            .map_err(|e| anyhow!("Channel error on GPU readback: {:?}", e))?
            .map_err(|e| anyhow!("Buffer map error on GPU readback: {:?}", e))?;

        {
            let view = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&view);
            for i in 0..out_len {
                out.data[i] = half::f16::from_f32(floats[i]);
            }
        }
        dense.staging_dn.unmap();
        Ok(())
    }

    fn expert_matmul(
        &self,
        layer_idx: usize,
        expert_id: u32,
        x:        TensorView<'_>,
        d_model:  usize,
        d_ff:     usize,
        out:      &mut TensorViewMut<'_>,
    ) -> Result<()> {
        let layer = u32::try_from(layer_idx)
            .map_err(|_| anyhow!("expert layer index {layer_idx} exceeds u32"))?;
        self.routed_expert_matmul(layer, expert_id, x, d_model, d_ff, out)
            .map_err(anyhow::Error::new)
    }
}

impl GpuBackend {
    /// Execute the already-constructed production Q4_0 WGSL pipeline over a
    /// small canonical block stream and return its f32 result. This seam is
    /// crate-private and is called only by `qualify-hybrid-q4-parity`: it owns
    /// ephemeral buffers, does not consult or mutate logical admission, does
    /// not touch the physical expert registry, and does not increment routed
    /// expert I/O counters.
    fn qualification_q4_0_matvec(
        &self,
        weights: &[u8],
        input: &[f32],
        rows: usize,
        columns: usize,
        w_block_off: usize,
        readback_timeout: Duration,
    ) -> std::result::Result<Vec<f32>, Q4ParityGpuError> {
        use crate::inference::{Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS};

        if let Some(detail) = self.device_loss.detail() {
            return Err(Q4ParityGpuError::new(
                Q4ParityGpuErrorKind::DeviceLost,
                format!("GPU device is lost before raw Q4_0 dispatch: {detail}"),
            ));
        }
        if rows == 0 || columns == 0 || !columns.is_multiple_of(Q4_0_BLOCK_ELEMS) {
            return Err(Q4ParityGpuError::new(
                Q4ParityGpuErrorKind::InvalidRequest,
                format!(
                    "invalid raw Q4_0 geometry rows={rows} columns={columns}; columns must be a non-zero multiple of {Q4_0_BLOCK_ELEMS}"
                ),
            ));
        }
        if input.len() != columns || input.iter().any(|value| !value.is_finite()) {
            return Err(Q4ParityGpuError::new(
                Q4ParityGpuErrorKind::InvalidRequest,
                format!(
                    "raw Q4_0 input has {} values for {columns} columns or contains a nonfinite value",
                    input.len()
                ),
            ));
        }
        let blocks_per_row = columns / Q4_0_BLOCK_ELEMS;
        let accessed_blocks = rows
            .checked_mul(blocks_per_row)
            .and_then(|count| w_block_off.checked_add(count))
            .ok_or_else(|| {
                Q4ParityGpuError::new(
                    Q4ParityGpuErrorKind::InvalidRequest,
                    "raw Q4_0 block geometry overflow",
                )
            })?;
        let required_weight_bytes = accessed_blocks
            .checked_mul(Q4_0_BLOCK_BYTES)
            .ok_or_else(|| {
                Q4ParityGpuError::new(
                    Q4ParityGpuErrorKind::InvalidRequest,
                    "raw Q4_0 byte length overflow",
                )
            })?;
        if weights.len() < required_weight_bytes
            || !weights.len().is_multiple_of(Q4_0_BLOCK_BYTES)
        {
            return Err(Q4ParityGpuError::new(
                Q4ParityGpuErrorKind::InvalidRequest,
                format!(
                    "raw Q4_0 weights have {} bytes, require at least {required_weight_bytes} bytes in complete {Q4_0_BLOCK_BYTES}-byte blocks",
                    weights.len()
                ),
            ));
        }
        let invalid_request = |detail| {
            Q4ParityGpuError::new(Q4ParityGpuErrorKind::InvalidRequest, detail)
        };
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| invalid_request(format!("raw Q4_0 rows {rows} exceed u32")))?;
        let columns_u32 = u32::try_from(columns)
            .map_err(|_| invalid_request(format!("raw Q4_0 columns {columns} exceed u32")))?;
        let block_off_u32 = u32::try_from(w_block_off).map_err(|_| {
            invalid_request(format!("raw Q4_0 w_block_off {w_block_off} exceeds u32"))
        })?;
        let weight_size = weights
            .len()
            .checked_add(3)
            .ok_or_else(|| invalid_request("raw Q4_0 padded weight size overflow".to_string()))?
            / 4
            * 4;
        let input_size = columns
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| invalid_request("raw Q4_0 input byte size overflow".to_string()))?;
        let output_size = rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| invalid_request("raw Q4_0 output byte size overflow".to_string()))?;
        let weight_size_u64 = u64::try_from(weight_size)
            .map_err(|_| invalid_request("raw Q4_0 weight size does not fit u64".to_string()))?;
        let input_size_u64 = u64::try_from(input_size)
            .map_err(|_| invalid_request("raw Q4_0 input size does not fit u64".to_string()))?;
        let output_size_u64 = u64::try_from(output_size)
            .map_err(|_| invalid_request("raw Q4_0 output size does not fit u64".to_string()))?;
        let pipeline = self.matmul_q4_0_pipeline.as_ref().ok_or_else(|| {
            Q4ParityGpuError::new(
                Q4ParityGpuErrorKind::ResourceUnavailable,
                format!(
                    "Q4_0 pipeline is unavailable in {:?} backend mode",
                    self.component_resources.plan().mode()
                ),
            )
        })?;

        // Capture qualification-created validation/OOM errors rather than
        // letting wgpu route them through the process-wide uncaptured handler.
        self.device
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = (|| -> std::result::Result<Vec<f32>, Q4ParityGpuError> {
            let weight_buf = create_startup_buffer(
                &self.device,
                "q4_parity_weights",
                weight_size_u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )
            .map_err(|error| {
                Q4ParityGpuError::new(Q4ParityGpuErrorKind::ResourceCreation, error.to_string())
            })?;
            let input_buf = create_startup_buffer(
                &self.device,
                "q4_parity_input",
                input_size_u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )
            .map_err(|error| {
                Q4ParityGpuError::new(Q4ParityGpuErrorKind::ResourceCreation, error.to_string())
            })?;
            let output_buf = create_startup_buffer(
                &self.device,
                "q4_parity_output",
                output_size_u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            )
            .map_err(|error| {
                Q4ParityGpuError::new(Q4ParityGpuErrorKind::ResourceCreation, error.to_string())
            })?;
            let staging = create_startup_buffer(
                &self.device,
                "q4_parity_staging",
                output_size_u64,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            )
            .map_err(|error| {
                Q4ParityGpuError::new(Q4ParityGpuErrorKind::ResourceCreation, error.to_string())
            })?;

            if weights.len() == weight_size {
                self.queue.write_buffer(&weight_buf, 0, weights);
            } else {
                let mut padded = Vec::with_capacity(weight_size);
                padded.extend_from_slice(weights);
                padded.resize(weight_size, 0);
                self.queue.write_buffer(&weight_buf, 0, &padded);
            }
            self.queue
                .write_buffer(&input_buf, 0, bytemuck::cast_slice(input));

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("q4_parity_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weight_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("q4_parity_encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("q4_parity_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_push_constants(
                    0,
                    bytemuck::bytes_of(&MatmulPushConstants {
                        m: rows_u32,
                        n: 1,
                        k: columns_u32,
                        w_block_off: block_off_u32,
                    }),
                );
                pass.dispatch_workgroups((rows_u32 + 63) / 64, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, output_size_u64);
            let _submission = self.queue.submit(Some(encoder.finish()));
            let slice = staging.slice(..output_size_u64);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |mapped| {
                let _ = tx.send(mapped);
            });
            wait_for_bounded_callback(
                &rx,
                readback_timeout,
                || {
                    self.device.poll(wgpu::Maintain::Poll);
                },
                || self.device_loss.detail(),
            )
            .map_err(|error| q4_parity_readback_error(error, readback_timeout))?;
            let output = {
                let view = slice.get_mapped_range();
                bytemuck::cast_slice::<u8, f32>(&view).to_vec()
            };
            staging.unmap();
            if output.iter().any(|value| !value.is_finite()) {
                return Err(Q4ParityGpuError::new(
                    Q4ParityGpuErrorKind::NonfiniteOutput,
                    "raw Q4_0 GPU output contains a nonfinite value",
                ));
            }
            Ok(output)
        })();
        let validation_error = pollster::block_on(self.device.pop_error_scope());
        let oom_error = pollster::block_on(self.device.pop_error_scope());
        if result.is_ok() {
            if let Some(error) = validation_error.or(oom_error) {
                return Err(Q4ParityGpuError::new(
                    Q4ParityGpuErrorKind::Validation,
                    format!("wgpu raw Q4_0 qualification error: {error}"),
                ));
            }
        }
        result
    }

    fn routed_expert_matmul(
        &self,
        layer: u32,
        expert_id: u32,
        x: TensorView<'_>,
        d_model: usize,
        d_ff: usize,
        out: &mut TensorViewMut<'_>,
    ) -> std::result::Result<(), GpuExpertDispatchError> {
        if let Some(detail) = self.device_loss.detail() {
            return Err(GpuExpertDispatchError::new(
                layer, expert_id, GpuExpertDispatchErrorKind::DeviceLost, detail,
            ));
        }

        let admission = match self.gpu_expert_cache.current_admission(expert_id) {
            Some(admission) => admission,
            None => {
                self.physical_expert_registry
                    .retire_logical_miss(expert_id);
                return Err(GpuExpertDispatchError::new(
                    layer,
                    expert_id,
                    GpuExpertDispatchErrorKind::ResidencyMiss,
                    "expert has no current logical GPU admission",
                ));
            }
        };
        let key = PhysicalExpertKey {
            expert_id,
            generation: admission.generation(),
        };
        match self.physical_expert_registry.lookup_current(key) {
            PhysicalRegistryLookup::Hit(entry) => {
                if !self
                    .gpu_expert_cache
                    .contains_generation(expert_id, key.generation)
                {
                    self.physical_expert_registry.retire_key_if_present(key);
                    return Err(GpuExpertDispatchError::new(
                        layer,
                        expert_id,
                        GpuExpertDispatchErrorKind::ResidencyMiss,
                        "logical GPU admission changed before physical dispatch",
                    ));
                }
                return self.expert_matmul_from_vram(layer, expert_id, &entry, x, out);
            }
            PhysicalRegistryLookup::Miss => {}
            PhysicalRegistryLookup::StaleRequester => {
                return Err(GpuExpertDispatchError::new(
                    layer,
                    expert_id,
                    GpuExpertDispatchErrorKind::ResidencyMiss,
                    "physical GPU residency has a newer logical generation",
                ));
            }
        }

        let resident = admission.resident();
        let plan = self
            .plan_expert_upload(resident.dtype(), resident.data(), d_model, d_ff)
            .map_err(|e| GpuExpertDispatchError::new(
                layer,
                expert_id,
                GpuExpertDispatchErrorKind::Upload,
                e.to_string(),
            ))?;
        let acquisition = self
            .physical_expert_registry
            .acquire_or_reserve(key, plan.device_bytes())
            .map_err(|e| GpuExpertDispatchError::new(
                layer,
                expert_id,
                GpuExpertDispatchErrorKind::PhysicalCapacity,
                e.to_string(),
            ))?;
        let entry = match acquisition {
            PhysicalRegistryAcquire::Hit(entry) => entry,
            PhysicalRegistryAcquire::StaleRequester => {
                return Err(GpuExpertDispatchError::new(
                    layer,
                    expert_id,
                    GpuExpertDispatchErrorKind::ResidencyMiss,
                    "physical GPU residency advanced before acquisition",
                ));
            }
            PhysicalRegistryAcquire::Install(mut permit) => {
                if !self
                    .gpu_expert_cache
                    .contains_generation(expert_id, key.generation)
                {
                    return Err(GpuExpertDispatchError::new(
                        layer,
                        expert_id,
                        GpuExpertDispatchErrorKind::ResidencyMiss,
                        "logical GPU admission changed before physical upload",
                    ));
                }
                let entry = match plan {
                    ExpertUploadPlan::F32(plan) => self.build_expert_entry(
                        key,
                        resident.data(),
                        d_model,
                        d_ff,
                        plan,
                        &mut permit,
                    ),
                    ExpertUploadPlan::Q4_0(plan) => self.build_expert_entry_q4_0(
                        key,
                        resident.data(),
                        d_model,
                        d_ff,
                        plan,
                        &mut permit,
                    ),
                }
                .map_err(|e| GpuExpertDispatchError::new(
                    layer,
                    expert_id,
                    GpuExpertDispatchErrorKind::Upload,
                    e.to_string(),
                ))?;
                if !self
                    .gpu_expert_cache
                    .contains_generation(expert_id, key.generation)
                {
                    drop(entry);
                    return Err(GpuExpertDispatchError::new(
                        layer,
                        expert_id,
                        GpuExpertDispatchErrorKind::ResidencyMiss,
                        "logical GPU admission changed during physical upload",
                    ));
                }
                permit.install(Arc::new(entry)).map_err(|detail| {
                    GpuExpertDispatchError::new(
                        layer,
                        expert_id,
                        GpuExpertDispatchErrorKind::RuntimeInvariant,
                        detail,
                    )
                })?
            }
        };
        // The generation may change between the post-upload check and the
        // registry install. Retire any just-installed stale entry before it
        // can become a dispatch result; an eviction after this point treats
        // this Arc as the already-in-flight activation it represents.
        if !self
            .gpu_expert_cache
            .contains_generation(expert_id, key.generation)
        {
            self.physical_expert_registry.retire_key_if_present(key);
            return Err(GpuExpertDispatchError::new(
                layer,
                expert_id,
                GpuExpertDispatchErrorKind::ResidencyMiss,
                "logical GPU admission changed before physical dispatch",
            ));
        }
        self.expert_matmul_from_vram(layer, expert_id, &entry, x, out)
    }

    fn gpu_expert_memory_snapshot(&self) -> GpuExpertMemorySnapshot {
        self.physical_expert_registry
            .snapshot(self.gpu_expert_cache.used_bytes())
    }

    fn compute_plane(&self) -> &str {
        &self.compute_plane
    }

    fn gpu_device_identity(&self) -> GpuDeviceIdentity {
        self.adapter_metadata.identity(&self.compute_plane)
    }

    fn gpu_expert_io_snapshot(&self) -> GpuExpertIoSnapshot {
        self.expert_io.snapshot()
    }

    fn gpu_physical_expert_residency(
        &self,
        expert_id: u32,
    ) -> Option<GpuPhysicalExpertResidency> {
        self.physical_expert_registry
            .residency_evidence(expert_id)
    }
}

// =====================================================================
// Candle CPU Fallback Backend
// =====================================================================

#[derive(Clone, Default)]
pub struct CandleBackend;

impl CandleBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CandleBackend {
    fn device_name(&self) -> &str {
        "cpu-fallback"
    }

    fn is_gpu(&self) -> bool {
        false
    }

    fn matmul_into(&self, a: TensorView, b: TensorView, out: &mut TensorViewMut) -> Result<()> {
        let m = a.rows;
        let k = a.cols;
        let n = b.cols;
        assert_eq!(b.rows, k);
        assert_eq!(out.rows, m);
        assert_eq!(out.cols, n);

        for val in out.data.iter_mut() {
            *val = half::f16::ZERO;
        }

        let tile_size = 32;
        for i_outer in (0..m).step_by(tile_size) {
            let i_end = (i_outer + tile_size).min(m);
            for k_outer in (0..k).step_by(tile_size) {
                let k_end = (k_outer + tile_size).min(k);
                for j_outer in (0..n).step_by(tile_size) {
                    let j_end = (j_outer + tile_size).min(n);

                    for i in i_outer..i_end {
                        let out_row_offset = i * n;
                        for k_inner in k_outer..k_end {
                            let a_val = a.data[i * k + k_inner].to_f32();
                            if a_val == 0.0 {
                                continue;
                            }
                            let b_row_offset = k_inner * n;
                            for j in j_outer..j_end {
                                let b_val = b.data[b_row_offset + j].to_f32();
                                let out_idx = out_row_offset + j;
                                let cur = out.data[out_idx].to_f32();
                                out.data[out_idx] = half::f16::from_f32(cur + a_val * b_val);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn swiglu_into(&self, gate: TensorView, up: TensorView, out: &mut TensorViewMut) -> Result<()> {
        let len = gate.data.len();
        assert_eq!(up.data.len(), len);
        assert_eq!(out.data.len(), len);

        // Apply the GPT-OSS gate clamp when active so this backend matches
        // both the GPU `swiglu.wgsl` path and the production CPU FFN kernel
        // (`kernels::scalar::swiglu_f32_clamped`). `None` is a no-op.
        let limit = crate::inference::swiglu_limit();
        for i in 0..len {
            let mut g = gate.data[i].to_f32();
            if let Some(l) = limit {
                g = g.clamp(-l, l);
            }
            let u = up.data[i].to_f32();
            let silu_g = g / (1.0 + (-g).exp());
            out.data[i] = half::f16::from_f32(silu_g * u);
        }
        Ok(())
    }

    fn softmax(&self, x: &mut TensorViewMut) -> Result<()> {
        let rows = x.rows;
        let cols = x.cols;
        for r in 0..rows {
            let row_slice = &mut x.data[r * cols..(r + 1) * cols];
            if row_slice.is_empty() {
                continue;
            }
            let mut maxv = f32::NEG_INFINITY;
            for &v in row_slice.iter() {
                let vf = v.to_f32();
                if vf > maxv {
                    maxv = vf;
                }
            }
            let mut sum = 0.0f32;
            for v in row_slice.iter_mut() {
                let vf = v.to_f32();
                let ev = (vf - maxv).exp();
                *v = half::f16::from_f32(ev);
                sum += ev;
            }
            if sum > 0.0 {
                for v in row_slice.iter_mut() {
                    *v = half::f16::from_f32(v.to_f32() / sum);
                }
            }
        }
        Ok(())
    }

    fn kv_cache_insert(
        &self,
        _layer: usize,
        _position: usize,
        _k: TensorView,
        _v: TensorView,
    ) -> Result<()> {
        // Managed on the CPU path directly in transformer.rs
        Ok(())
    }

    fn kv_attend(
        &self,
        _layer: usize,
        _q: TensorView,
        _seq_len: usize,
        _out: &mut TensorViewMut,
    ) -> Result<()> {
        // Managed on the CPU path directly in transformer.rs
        Ok(())
    }

    fn expert_matmul(
        &self,
        _layer_idx: usize,
        _expert_id: u32,
        _x:        TensorView<'_>,
        _d_model:  usize,
        _d_ff:     usize,
        _out:      &mut TensorViewMut<'_>,
    ) -> Result<()> {
        anyhow::bail!("expert_matmul should not be called on CPU backend; use direct NVMe streaming path instead")
    }
}

/// Hardware-independent routed-expert GPU boundary used only by unit tests.
#[cfg(test)]
#[derive(Clone)]
pub(crate) struct TestGpuBackend {
    cpu: CandleBackend,
    outcome: TestGpuExpertOutcome,
    expert_calls: Arc<AtomicU64>,
}

#[cfg(test)]
#[derive(Clone)]
enum TestGpuExpertOutcome {
    Success(f32),
    Failure(GpuExpertDispatchErrorKind),
}

#[cfg(test)]
impl TestGpuBackend {
    pub(crate) fn success(value: f32) -> Self {
        Self {
            cpu: CandleBackend::new(),
            outcome: TestGpuExpertOutcome::Success(value),
            expert_calls: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(crate) fn failure(kind: GpuExpertDispatchErrorKind) -> Self {
        Self {
            cpu: CandleBackend::new(),
            outcome: TestGpuExpertOutcome::Failure(kind),
            expert_calls: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(crate) fn expert_calls(&self) -> u64 {
        self.expert_calls.load(Ordering::Relaxed)
    }

    fn routed_expert_matmul(
        &self,
        layer: u32,
        expert_id: u32,
        out: &mut TensorViewMut<'_>,
    ) -> std::result::Result<(), GpuExpertDispatchError> {
        self.expert_calls.fetch_add(1, Ordering::Relaxed);
        match self.outcome {
            TestGpuExpertOutcome::Success(value) => {
                out.data.fill(half::f16::from_f32(value));
                Ok(())
            }
            TestGpuExpertOutcome::Failure(kind) => Err(GpuExpertDispatchError::new(
                layer, expert_id, kind, format!("injected {kind} failure"),
            )),
        }
    }
}

// =====================================================================
// BackendBox Dispatch Enum (Zero-cost dispatch, no dyn/vtable)
// =====================================================================

pub enum BackendBox {
    Gpu(GpuBackend),
    Cpu(CandleBackend),
    #[cfg(test)]
    TestGpu(TestGpuBackend),
}

impl BackendBox {
    pub fn compute_plane(&self) -> &str {
        match self {
            BackendBox::Gpu(gpu) => gpu.compute_plane(),
            BackendBox::Cpu(_) => "cpu-fallback",
            #[cfg(test)]
            BackendBox::TestGpu(_) => "test-gpu",
        }
    }

    /// Exact PR4 routed-expert physical-memory ledger when this is a
    /// production GPU backend. CPU and hardware-independent test backends do
    /// not own physical wgpu expert allocations.
    pub fn gpu_expert_memory_snapshot(&self) -> Option<GpuExpertMemorySnapshot> {
        match self {
            Self::Gpu(gpu) => Some(gpu.gpu_expert_memory_snapshot()),
            Self::Cpu(_) => None,
            #[cfg(test)]
            Self::TestGpu(_) => None,
        }
    }

    pub fn gpu_device_identity(&self) -> Option<GpuDeviceIdentity> {
        match self {
            Self::Gpu(gpu) => Some(gpu.gpu_device_identity()),
            Self::Cpu(_) => None,
            #[cfg(test)]
            Self::TestGpu(_) => None,
        }
    }

    pub fn gpu_expert_io_snapshot(&self) -> Option<GpuExpertIoSnapshot> {
        match self {
            Self::Gpu(gpu) => Some(gpu.gpu_expert_io_snapshot()),
            Self::Cpu(_) => None,
            #[cfg(test)]
            Self::TestGpu(_) => None,
        }
    }

    pub(crate) fn gpu_physical_expert_residency(
        &self,
        expert_id: u32,
    ) -> Option<GpuPhysicalExpertResidency> {
        match self {
            Self::Gpu(gpu) => gpu.gpu_physical_expert_residency(expert_id),
            Self::Cpu(_) => None,
            #[cfg(test)]
            Self::TestGpu(_) => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qualification_q4_0_matvec(
        &self,
        weights: &[u8],
        input: &[f32],
        rows: usize,
        columns: usize,
        w_block_off: usize,
        readback_timeout: Duration,
    ) -> std::result::Result<Vec<f32>, Q4ParityGpuError> {
        match self {
            Self::Gpu(gpu) => gpu.qualification_q4_0_matvec(
                weights,
                input,
                rows,
                columns,
                w_block_off,
                readback_timeout,
            ),
            Self::Cpu(_) => Err(Q4ParityGpuError::new(
                Q4ParityGpuErrorKind::ResourceUnavailable,
                "raw Q4_0 qualification requires the authoritative production GPU backend",
            )),
            #[cfg(test)]
            Self::TestGpu(_) => Err(Q4ParityGpuError::new(
                Q4ParityGpuErrorKind::ResourceUnavailable,
                "raw Q4_0 qualification cannot run against the hardware-independent test backend",
            )),
        }
    }

    /// Typed dispatch for real-model routed experts. The legacy trait method
    /// maps this to anyhow only for synthetic compatibility diagnostics.
    #[allow(clippy::too_many_arguments)]
    pub fn routed_expert_matmul(
        &self,
        layer: u32,
        expert_id: u32,
        x: TensorView<'_>,
        d_model: usize,
        d_ff: usize,
        out: &mut TensorViewMut<'_>,
    ) -> std::result::Result<(), GpuExpertDispatchError> {
        match self {
            Self::Gpu(gpu) => gpu.routed_expert_matmul(layer, expert_id, x, d_model, d_ff, out),
            Self::Cpu(_) => Err(GpuExpertDispatchError::new(
                layer,
                expert_id,
                GpuExpertDispatchErrorKind::RuntimeInvariant,
                "resolved GPU routed-expert plan selected a CPU backend",
            )),
            #[cfg(test)]
            Self::TestGpu(gpu) => gpu.routed_expert_matmul(layer, expert_id, out),
        }
    }
}

impl Backend for BackendBox {
    fn device_name(&self) -> &str {
        match self {
            BackendBox::Gpu(gpu) => gpu.device_name(),
            BackendBox::Cpu(cpu) => cpu.device_name(),
            #[cfg(test)]
            BackendBox::TestGpu(_) => "test-gpu",
        }
    }

    fn is_gpu(&self) -> bool {
        match self {
            BackendBox::Gpu(gpu) => gpu.is_gpu(),
            BackendBox::Cpu(cpu) => cpu.is_gpu(),
            #[cfg(test)]
            BackendBox::TestGpu(_) => true,
        }
    }

    fn matmul_into(&self, a: TensorView, b: TensorView, out: &mut TensorViewMut) -> Result<()> {
        match self {
            BackendBox::Gpu(gpu) => gpu.matmul_into(a, b, out),
            BackendBox::Cpu(cpu) => cpu.matmul_into(a, b, out),
            #[cfg(test)]
            BackendBox::TestGpu(gpu) => gpu.cpu.matmul_into(a, b, out),
        }
    }

    fn swiglu_into(&self, gate: TensorView, up: TensorView, out: &mut TensorViewMut) -> Result<()> {
        match self {
            BackendBox::Gpu(gpu) => gpu.swiglu_into(gate, up, out),
            BackendBox::Cpu(cpu) => cpu.swiglu_into(gate, up, out),
            #[cfg(test)]
            BackendBox::TestGpu(gpu) => gpu.cpu.swiglu_into(gate, up, out),
        }
    }

    fn softmax(&self, x: &mut TensorViewMut) -> Result<()> {
        match self {
            BackendBox::Gpu(gpu) => gpu.softmax(x),
            BackendBox::Cpu(cpu) => cpu.softmax(x),
            #[cfg(test)]
            BackendBox::TestGpu(gpu) => gpu.cpu.softmax(x),
        }
    }

    fn kv_cache_insert(
        &self,
        layer: usize,
        position: usize,
        k: TensorView,
        v: TensorView,
    ) -> Result<()> {
        match self {
            BackendBox::Gpu(gpu) => gpu.kv_cache_insert(layer, position, k, v),
            BackendBox::Cpu(cpu) => cpu.kv_cache_insert(layer, position, k, v),
            #[cfg(test)]
            BackendBox::TestGpu(gpu) => gpu.cpu.kv_cache_insert(layer, position, k, v),
        }
    }

    fn kv_attend(
        &self,
        layer: usize,
        q: TensorView,
        seq_len: usize,
        out: &mut TensorViewMut,
    ) -> Result<()> {
        match self {
            BackendBox::Gpu(gpu) => gpu.kv_attend(layer, q, seq_len, out),
            BackendBox::Cpu(cpu) => cpu.kv_attend(layer, q, seq_len, out),
            #[cfg(test)]
            BackendBox::TestGpu(gpu) => gpu.cpu.kv_attend(layer, q, seq_len, out),
        }
    }

    fn expert_matmul(
        &self,
        layer_idx: usize,
        expert_id: u32,
        x:        TensorView<'_>,
        d_model:  usize,
        d_ff:     usize,
        out:      &mut TensorViewMut<'_>,
    ) -> Result<()> {
        let layer = u32::try_from(layer_idx)
            .map_err(|_| anyhow!("expert layer index {layer_idx} exceeds u32"))?;
        self.routed_expert_matmul(layer, expert_id, x, d_model, d_ff, out)
            .map_err(anyhow::Error::new)
    }
}

// =====================================================================
// Operator-facing ComputeOffload Enum
// =====================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeOffload {
    Cpu,
    Gpu,
    /// Prefer GPU but fall back to CPU if GPU initialization fails. Unlike
    /// an explicit `Gpu` request (which fails closed), `Auto` treats GPU as
    /// best-effort and records a fallback event when it lands on CPU. Under
    /// strict attention numerics `Auto` resolves to CPU with a recorded
    /// reason (GPU on-device softmax cannot be numerically validated).
    Auto,
    /// **Explicitly named hybrid mode**: attention runs on the *checked
    /// CPU* path (strict numerics validated per softmax row) while
    /// routed-expert FFN compute is offloaded to the GPU. This is the only
    /// mode in which a GPU backend is installed under strict attention
    /// numerics — the split is logged at startup and exposed through
    /// per-component backend metrics, never silently reported as full
    /// GPU execution.
    Hybrid,
}

impl Default for ComputeOffload {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Backend the runtime actually resolved to after (optionally) attempting
/// GPU initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedBackend {
    Cpu,
    Gpu,
    /// CPU (checked) attention + GPU expert compute — the explicitly
    /// configured [`ComputeOffload::Hybrid`] split.
    HybridCpuAttentionGpuExperts,
}

/// Outcome of reconciling an operator's requested [`ComputeOffload`] with
/// the result of GPU initialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendResolution {
    pub requested: ComputeOffload,
    pub resolved: ResolvedBackend,
    /// True only when GPU was attempted, failed, and `Auto` demoted the run
    /// to CPU. An explicit `Gpu`/`Hybrid` request never produces a silent
    /// fallback — it errors instead.
    pub fallback_occurred: bool,
    /// Human-readable reason recorded whenever the resolution differs from
    /// a naive reading of the request (e.g. `auto` resolving to CPU under
    /// strict attention numerics, or after a failed GPU init).
    pub reason: Option<String>,
}

/// Concrete execution plane selected for one transformer component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPlane {
    Cpu,
    Gpu,
}

impl ExecutionPlane {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

/// Runtime inputs that determine whether the current routed-expert GPU
/// kernels can execute a resident expert without a deterministic CPU bypass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutedExpertGpuSpec {
    pub dtype: crate::inference::WeightDtype,
    pub d_model: usize,
    pub d_ff: usize,
}

/// Pure compatibility failure returned by [`routed_expert_gpu_compatibility`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutedExpertGpuIncompatibility {
    UnsupportedDtype {
        dtype: crate::inference::WeightDtype,
    },
    ShapeExceedsWorkspace {
        d_model: usize,
        d_ff: usize,
        max: usize,
    },
    MisalignedQ4_0 {
        d_model: usize,
        d_ff: usize,
        block_elems: usize,
    },
    F32ProjectionSizeOverflow {
        d_model: usize,
        d_ff: usize,
    },
    F32StorageOffsetMisaligned {
        projection_bytes: usize,
        required_alignment: u64,
    },
}

impl fmt::Display for RoutedExpertGpuIncompatibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDtype { dtype } => write!(
                f,
                "routed-expert dtype {} is unsupported by the GPU expert path; supported dtypes \
                 are f32 and block-aligned q4_0",
                dtype.as_str()
            ),
            Self::ShapeExceedsWorkspace { d_model, d_ff, max } => write!(
                f,
                "routed-expert GPU workspace supports d_model and d_ff up to {max}; got \
                 d_model={d_model}, d_ff={d_ff}"
            ),
            Self::MisalignedQ4_0 {
                d_model,
                d_ff,
                block_elems,
            } => write!(
                f,
                "routed-expert q4_0 geometry requires d_model and d_ff to be multiples of \
                 {block_elems}; got d_model={d_model}, d_ff={d_ff}"
            ),
            Self::F32ProjectionSizeOverflow { d_model, d_ff } => write!(
                f,
                "routed-expert f32 projection size overflows addressable storage for \
                 d_model={d_model}, d_ff={d_ff}"
            ),
            Self::F32StorageOffsetMisaligned {
                projection_bytes,
                required_alignment,
            } => write!(
                f,
                "routed-expert f32 projection size {projection_bytes} bytes does not satisfy \
                 the selected GPU's min_storage_buffer_offset_alignment={required_alignment}"
            ),
        }
    }
}

impl std::error::Error for RoutedExpertGpuIncompatibility {}

/// Authoritative device-independent routed-expert GPU compatibility rule
/// shared by resolution and execution. Device-specific limits are validated
/// separately during backend construction. Both checks must stay aligned with
/// the formats implemented by `GpuBackend::expert_matmul`.
pub fn routed_expert_gpu_compatibility(
    spec: RoutedExpertGpuSpec,
) -> std::result::Result<(), RoutedExpertGpuIncompatibility> {
    use crate::inference::{WeightDtype, Q4_0_BLOCK_ELEMS};

    if spec.d_model > MAX_EXPERT_D_FF || spec.d_ff > MAX_EXPERT_D_FF {
        return Err(RoutedExpertGpuIncompatibility::ShapeExceedsWorkspace {
            d_model: spec.d_model,
            d_ff: spec.d_ff,
            max: MAX_EXPERT_D_FF,
        });
    }

    match spec.dtype {
        WeightDtype::F32 => Ok(()),
        WeightDtype::Q4_0
            if spec.d_model.is_multiple_of(Q4_0_BLOCK_ELEMS)
                && spec.d_ff.is_multiple_of(Q4_0_BLOCK_ELEMS) =>
        {
            Ok(())
        }
        WeightDtype::Q4_0 => Err(RoutedExpertGpuIncompatibility::MisalignedQ4_0 {
            d_model: spec.d_model,
            d_ff: spec.d_ff,
            block_elems: Q4_0_BLOCK_ELEMS,
        }),
        dtype => Err(RoutedExpertGpuIncompatibility::UnsupportedDtype { dtype }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct F32ExpertProjectionLayout {
    projection_bytes: usize,
    projection_offset: u64,
    required_bytes: usize,
}

/// Checked F32 projection layout shared by startup compatibility validation
/// and the defensive upload path.
fn f32_expert_projection_layout(
    spec: RoutedExpertGpuSpec,
    required_alignment: u64,
) -> std::result::Result<F32ExpertProjectionLayout, RoutedExpertGpuIncompatibility> {
    let overflow = || RoutedExpertGpuIncompatibility::F32ProjectionSizeOverflow {
        d_model: spec.d_model,
        d_ff: spec.d_ff,
    };
    let projection_bytes = spec
        .d_ff
        .checked_mul(spec.d_model)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(overflow)?;
    let down_offset_bytes = projection_bytes.checked_mul(2).ok_or_else(overflow)?;
    let required_bytes = projection_bytes.checked_mul(3).ok_or_else(overflow)?;
    let projection_offset = u64::try_from(projection_bytes).map_err(|_| overflow())?;
    let down_offset = u64::try_from(down_offset_bytes).map_err(|_| overflow())?;
    let required_alignment = required_alignment.max(1);
    if !projection_offset.is_multiple_of(required_alignment)
        || !down_offset.is_multiple_of(required_alignment)
    {
        return Err(
            RoutedExpertGpuIncompatibility::F32StorageOffsetMisaligned {
                projection_bytes,
                required_alignment,
            },
        );
    }
    Ok(F32ExpertProjectionLayout {
        projection_bytes,
        projection_offset,
        required_bytes,
    })
}

/// Convert a validated F32 projection layout into the exact device upload
/// plan. Host payloads may include storage padding beyond the three logical
/// projections; that padding is neither allocated nor copied to the GPU.
fn f32_expert_upload_plan(
    layout: F32ExpertProjectionLayout,
    host_bytes: usize,
    d_model: usize,
    d_ff: usize,
) -> anyhow::Result<F32ExpertUploadPlan> {
    anyhow::ensure!(
        host_bytes >= layout.required_bytes,
        "expert weight buffer too small: got {} bytes, need {} (3 × d_ff={} × d_model={} × 4)",
        host_bytes,
        layout.required_bytes,
        d_ff,
        d_model
    );
    let device_bytes = u64::try_from(layout.required_bytes)
        .map_err(|_| anyhow!("F32 expert buffer length does not fit u64"))?;
    Ok(F32ExpertUploadPlan {
        device_bytes,
        projection_offset: layout.projection_offset,
    })
}

fn routed_expert_gpu_device_compatibility(
    spec: RoutedExpertGpuSpec,
    min_storage_buffer_offset_alignment: u64,
) -> std::result::Result<(), RoutedExpertGpuIncompatibility> {
    if spec.dtype == crate::inference::WeightDtype::F32 {
        f32_expert_projection_layout(spec, min_storage_buffer_offset_alignment)?;
    }
    Ok(())
}

/// Stable, process-local identity for one resolved execution context.
///
/// This is deliberately not a pointer address: it is safe to include in
/// startup diagnostics and lets tests prove that the plan and runtime consume
/// the same context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionContextId(u64);

impl fmt::Display for ExecutionContextId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

static NEXT_EXECUTION_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_execution_context_id() -> ExecutionContextId {
    let id = NEXT_EXECUTION_CONTEXT_ID.fetch_add(1, Ordering::Relaxed);
    assert_ne!(id, 0, "execution context id space exhausted");
    ExecutionContextId(id)
}

/// Immutable source of truth for requested mode and resolved component planes.
///
/// Requested mode is operator intent; it is not an execution report. Runtime
/// consumers and reporting must use these resolved planes from the authoritative
/// [`ExecutionContext`] instead of re-deriving them from configuration flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedExecutionPlan {
    requested: ComputeOffload,
    resolved: ResolvedBackend,
    context_id: ExecutionContextId,
    embeddings: ExecutionPlane,
    lm_head: ExecutionPlane,
    dense_projections: ExecutionPlane,
    attention: ExecutionPlane,
    kv: ExecutionPlane,
    router: ExecutionPlane,
    routed_experts: ExecutionPlane,
    routed_expert_gpu_spec: RoutedExpertGpuSpec,
    fallback_occurred: bool,
    reason: Option<String>,
}

impl ResolvedExecutionPlan {
    fn from_resolution(
        resolution: BackendResolution,
        context_id: ExecutionContextId,
        gpu_expert_cache_available: bool,
        routed_expert_gpu_spec: RoutedExpertGpuSpec,
        routed_expert_gpu_compatible: bool,
    ) -> Self {
        let cpu = ExecutionPlane::Cpu;
        let (attention, kv, routed_experts) = match resolution.resolved {
            ResolvedBackend::Cpu => (cpu, cpu, cpu),
            ResolvedBackend::Gpu => (
                ExecutionPlane::Gpu,
                ExecutionPlane::Gpu,
                if gpu_expert_cache_available && routed_expert_gpu_compatible {
                    ExecutionPlane::Gpu
                } else {
                    cpu
                },
            ),
            ResolvedBackend::HybridCpuAttentionGpuExperts => (
                cpu,
                cpu,
                if gpu_expert_cache_available && routed_expert_gpu_compatible {
                    ExecutionPlane::Gpu
                } else {
                    cpu
                },
            ),
        };
        Self {
            requested: resolution.requested,
            resolved: resolution.resolved,
            context_id,
            embeddings: cpu,
            lm_head: cpu,
            dense_projections: cpu,
            attention,
            kv,
            router: cpu,
            routed_experts,
            routed_expert_gpu_spec,
            fallback_occurred: resolution.fallback_occurred,
            reason: resolution.reason,
        }
    }

    pub const fn requested(&self) -> ComputeOffload {
        self.requested
    }

    pub const fn resolved(&self) -> ResolvedBackend {
        self.resolved
    }

    pub const fn context_id(&self) -> ExecutionContextId {
        self.context_id
    }

    pub const fn embeddings(&self) -> ExecutionPlane {
        self.embeddings
    }

    pub const fn lm_head(&self) -> ExecutionPlane {
        self.lm_head
    }

    pub const fn dense_projections(&self) -> ExecutionPlane {
        self.dense_projections
    }

    pub const fn attention(&self) -> ExecutionPlane {
        self.attention
    }

    pub const fn kv(&self) -> ExecutionPlane {
        self.kv
    }

    pub const fn router(&self) -> ExecutionPlane {
        self.router
    }

    pub const fn routed_experts(&self) -> ExecutionPlane {
        self.routed_experts
    }

    pub const fn routed_expert_gpu_spec(&self) -> RoutedExpertGpuSpec {
        self.routed_expert_gpu_spec
    }

    pub const fn fallback_occurred(&self) -> bool {
        self.fallback_occurred
    }

    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }

    /// Stable metric labels for every explicitly modelled component.
    pub const fn component_planes(&self) -> [(&'static str, ExecutionPlane); 7] {
        [
            ("embeddings", self.embeddings),
            ("lm_head", self.lm_head),
            ("dense_projections", self.dense_projections),
            ("attention", self.attention),
            ("kv", self.kv),
            ("router", self.router),
            // Keep the existing public metric label for compatibility; the
            // typed plan calls the component `routed_experts` explicitly.
            ("experts", self.routed_experts),
        ]
    }

    /// GPU construction capability derived from this authoritative component
    /// plan. Hybrid owns only routed-expert resources; the legacy GPU plan
    /// retains the full resource set.
    pub const fn gpu_backend_mode(&self) -> Option<GpuBackendMode> {
        match (self.attention, self.kv, self.routed_experts) {
            (ExecutionPlane::Cpu, ExecutionPlane::Cpu, ExecutionPlane::Gpu) => {
                Some(GpuBackendMode::RoutedExpertsOnly)
            }
            (ExecutionPlane::Gpu, ExecutionPlane::Gpu, _) => Some(GpuBackendMode::Full),
            _ => None,
        }
    }
}

/// Hardware-independent startup resource plan for one authoritative resolved
/// execution plan. Tests use this seam to prove Hybrid allocates no dense,
/// attention, or KV resources without requiring a live adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GpuResourcePlan {
    mode: GpuBackendMode,
    routed_expert_dtype: crate::inference::WeightDtype,
    kv_allocation_bytes: u64,
    dense_allocation_bytes: u64,
    expert_workspace_allocation_bytes: u64,
}

impl GpuResourcePlan {
    fn from_execution_plan(
        plan: &ResolvedExecutionPlan,
        geometry: GpuBackendGeometry,
    ) -> std::result::Result<Self, BackendResolutionError> {
        let mode = plan.gpu_backend_mode().ok_or_else(|| {
            BackendResolutionError::InvalidGeometry {
                detail: "CPU-only execution plan cannot construct a GPU resource plan".to_string(),
            }
        })?;
        geometry.validate_for_mode(mode)?;
        let full = mode == GpuBackendMode::Full;
        let routed_expert_dtype = plan.routed_expert_gpu_spec().dtype;
        let kv_allocation_bytes = if full {
            geometry.kv_allocation_bytes()?
        } else {
            0
        };
        let dense_allocation_bytes = if full {
            DENSE_BUFFER_BYTES
                .checked_mul(5)
                .ok_or_else(|| BackendResolutionError::InvalidGeometry {
                    detail: "dense GPU startup allocation byte total overflow".to_string(),
                })?
        } else {
            0
        };
        let workspace_buffer_bytes = (MAX_EXPERT_D_FF * std::mem::size_of::<f32>()) as u64;
        let expert_workspace_allocation_bytes = expert_workspace_device_bytes(
            workspace_buffer_bytes,
            EXPERT_WORKSPACE_POOL,
        )
        .ok_or_else(|| BackendResolutionError::InvalidGeometry {
            detail: "routed-expert workspace byte total overflow".to_string(),
        })?;
        Ok(Self {
            mode,
            routed_expert_dtype,
            kv_allocation_bytes,
            dense_allocation_bytes,
            expert_workspace_allocation_bytes,
        })
    }

    const fn mode(self) -> GpuBackendMode {
        self.mode
    }

    const fn kv_allocation_bytes(self) -> u64 {
        self.kv_allocation_bytes
    }

    const fn dense_allocation_bytes(self) -> u64 {
        self.dense_allocation_bytes
    }

    const fn expert_workspace_allocation_bytes(self) -> u64 {
        self.expert_workspace_allocation_bytes
    }

    const fn constructs_attention_resources(self) -> bool {
        matches!(self.mode, GpuBackendMode::Full)
    }

    const fn constructs_dense_resources(self) -> bool {
        matches!(self.mode, GpuBackendMode::Full)
    }

    const fn constructs_f32_matmul_pipeline(self) -> bool {
        matches!(self.mode, GpuBackendMode::Full)
            || matches!(self.routed_expert_dtype, crate::inference::WeightDtype::F32)
    }

    const fn constructs_q4_0_matmul_pipeline(self) -> bool {
        matches!(self.mode, GpuBackendMode::Full)
            || matches!(self.routed_expert_dtype, crate::inference::WeightDtype::Q4_0)
    }

    const fn has_capability(self, capability: GpuCapability) -> bool {
        match capability {
            GpuCapability::RoutedExperts => true,
            GpuCapability::Dense | GpuCapability::Attention => {
                matches!(self.mode, GpuBackendMode::Full)
            }
            GpuCapability::Kv => self.kv_allocation_bytes > 0,
        }
    }

    fn require_capability(
        self,
        capability: GpuCapability,
    ) -> std::result::Result<(), GpuCapabilityUnavailable> {
        if self.has_capability(capability) {
            Ok(())
        } else {
            Err(GpuCapabilityUnavailable {
                mode: self.mode,
                capability,
            })
        }
    }
}

/// The one authoritative backend context for a resolved runtime.
///
/// A hybrid context owns one CPU backend and exactly one GPU backend. Component
/// accessors select between those immutable members using the resolved plan, so
/// neither the model nor the engine can independently construct or choose a
/// different backend. The expert cache is owned here for the same reason: the
/// engine attaches the exact cache used to construct the GPU backend.
pub struct ExecutionContext {
    plan: ResolvedExecutionPlan,
    cpu_backend: Arc<BackendBox>,
    gpu_backend: Option<Arc<BackendBox>>,
    gpu_expert_cache: Arc<crate::expert_cache::GpuExpertCache>,
}

impl ExecutionContext {
    fn new(
        plan: ResolvedExecutionPlan,
        gpu_backend: Option<Arc<BackendBox>>,
        gpu_expert_cache: Arc<crate::expert_cache::GpuExpertCache>,
    ) -> Self {
        assert_eq!(
            plan.component_planes()
                .iter()
                .any(|(_, plane)| *plane == ExecutionPlane::Gpu),
            gpu_backend.is_some(),
            "resolved component planes and GPU backend ownership disagree: {plan:?}"
        );
        assert!(
            plan.routed_experts() != ExecutionPlane::Gpu || gpu_expert_cache.capacity_bytes() > 0,
            "GPU routed-expert plane requires non-zero expert-weight capacity"
        );
        Self {
            plan,
            cpu_backend: Arc::new(BackendBox::Cpu(CandleBackend::new())),
            gpu_backend,
            gpu_expert_cache,
        }
    }

    pub fn plan(&self) -> &ResolvedExecutionPlan {
        &self.plan
    }

    pub const fn id(&self) -> ExecutionContextId {
        self.plan.context_id()
    }

    fn backend_for(&self, plane: ExecutionPlane) -> &Arc<BackendBox> {
        match plane {
            ExecutionPlane::Cpu => &self.cpu_backend,
            ExecutionPlane::Gpu => self
                .gpu_backend
                .as_ref()
                .expect("resolved GPU plane must own a GPU backend"),
        }
    }

    pub fn attention_backend(&self) -> &Arc<BackendBox> {
        self.backend_for(self.plan.attention())
    }

    pub fn routed_expert_backend(&self) -> &Arc<BackendBox> {
        self.backend_for(self.plan.routed_experts())
    }

    pub fn gpu_expert_cache(&self) -> &Arc<crate::expert_cache::GpuExpertCache> {
        &self.gpu_expert_cache
    }

    /// Narrow snapshot seam for PR5. Existing compatibility telemetry remains
    /// logical-admission based; this API exposes exact MER-owned physical
    /// expert and workspace bytes without log parsing.
    pub fn gpu_expert_memory_snapshot(&self) -> Option<GpuExpertMemorySnapshot> {
        self.gpu_backend
            .as_ref()
            .and_then(|backend| backend.gpu_expert_memory_snapshot())
    }

    pub fn gpu_device_identity(&self) -> Option<GpuDeviceIdentity> {
        self.gpu_backend
            .as_ref()
            .and_then(|backend| backend.gpu_device_identity())
    }

    pub fn gpu_expert_io_snapshot(&self) -> Option<GpuExpertIoSnapshot> {
        self.gpu_backend
            .as_ref()
            .and_then(|backend| backend.gpu_expert_io_snapshot())
    }

    pub fn primary_backend(&self) -> &Arc<BackendBox> {
        self.gpu_backend.as_ref().unwrap_or(&self.cpu_backend)
    }
}

impl fmt::Debug for ExecutionContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("plan", &self.plan)
            .field("device", &self.primary_backend().device_name())
            .finish()
    }
}

/// Geometry and policy inputs that must be valid before GPU initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuBackendGeometry {
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub v_head_dim: usize,
    pub q4_truncation_tolerance: usize,
}

impl GpuBackendGeometry {
    fn validate_for_mode(
        &self,
        mode: GpuBackendMode,
    ) -> std::result::Result<(), BackendResolutionError> {
        if mode == GpuBackendMode::RoutedExpertsOnly {
            return Ok(());
        }
        if self.num_layers == 0
            || self.max_seq_len == 0
            || self.num_heads == 0
            || self.head_dim == 0
        {
            return Err(BackendResolutionError::InvalidGeometry {
                detail: format!(
                    "GPU geometry requires non-zero num_layers, max_seq_len, num_heads, and \
                     head_dim (got {}, {}, {}, {})",
                    self.num_layers, self.max_seq_len, self.num_heads, self.head_dim
                ),
            });
        }
        let kv_heads = if self.num_kv_heads == 0 {
            self.num_heads
        } else {
            self.num_kv_heads
        };
        if self.num_heads % kv_heads != 0 {
            return Err(BackendResolutionError::InvalidGeometry {
                detail: format!(
                    "GPU geometry requires num_kv_heads ({kv_heads}) to divide num_heads ({})",
                    self.num_heads
                ),
            });
        }
        self.kv_allocation_bytes()?;
        Ok(())
    }

    fn kv_allocation_bytes(&self) -> std::result::Result<u64, BackendResolutionError> {
        let kv_heads = if self.num_kv_heads == 0 {
            self.num_heads
        } else {
            self.num_kv_heads
        };
        let v_head_dim = if self.v_head_dim == 0 {
            self.head_dim
        } else {
            self.v_head_dim
        };
        kv_heads
            .checked_mul(self.head_dim)
            .zip(kv_heads.checked_mul(v_head_dim))
            .and_then(|(k, v)| k.checked_add(v))
            .and_then(|per_token| per_token.checked_mul(self.max_seq_len))
            .and_then(|per_layer| per_layer.checked_mul(self.num_layers))
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| BackendResolutionError::InvalidGeometry {
                detail: "GPU KV geometry overflows addressable memory".to_string(),
            })
    }
}

/// Error returned when a requested compute backend cannot be honored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendResolutionError {
    /// Invalid GPU sizing or attention geometry detected before adapter or
    /// device initialization.
    InvalidGeometry { detail: String },
    /// Hybrid selects GPU routed experts, but no non-zero expert-weight capacity
    /// was configured to make that plane executable.
    RoutedExpertGpuCacheRequired,
    /// Explicit Hybrid requested GPU routed experts, but the current expert
    /// dtype or geometry cannot execute through the GPU expert kernels.
    RoutedExpertGpuIncompatible {
        requested: ComputeOffload,
        incompatibility: RoutedExpertGpuIncompatibility,
    },
    /// An explicit `gpu`/`hybrid` request but GPU initialization failed.
    GpuUnavailable {
        requested: ComputeOffload,
        detail: String,
    },
    /// `compute_offload = "gpu"` combined with strict attention numerics:
    /// GPU attention performs its softmax on-device where non-finite rows
    /// cannot be detected, so strict GPU-attention inference is unsupported
    /// until GPU numerical detection exists.
    StrictGpuUnsupported,
    /// `compute_offload = "hybrid"` combined with
    /// `allow_nonfinite_attention_fallback = true`: hybrid pins attention
    /// to the checked CPU path, which is incompatible with the legacy
    /// uniform-fallback attention policy.
    HybridRequiresStrictAttention,
}

impl std::fmt::Display for BackendResolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendResolutionError::InvalidGeometry { detail } => {
                write!(f, "invalid GPU backend geometry: {detail}")
            }
            BackendResolutionError::RoutedExpertGpuCacheRequired => write!(
                f,
                "compute_offload = \"hybrid\" requires gpu_cache.enabled = true and \
                 gpu_cache.vram_capacity_mb > 0 so routed experts can execute on GPU"
            ),
            BackendResolutionError::RoutedExpertGpuIncompatible {
                requested,
                incompatibility,
            } => write!(
                f,
                "compute_offload = {requested:?} requires GPU routed experts, but the current \
                 expert path is incompatible: {incompatibility}"
            ),
            BackendResolutionError::GpuUnavailable { requested, detail } => write!(
                f,
                "compute_offload = {requested:?} was requested but GPU initialization failed: \
                 {detail}; set compute_offload = \"auto\" to allow CPU fallback, or \"cpu\" to \
                 run on CPU"
            ),
            BackendResolutionError::StrictGpuUnsupported => write!(
                f,
                "compute_offload = \"gpu\" is unsupported under strict attention numerics: the \
                 GPU attention kernels compute softmax on-device where non-finite rows cannot \
                 be detected or validated. Use compute_offload = \"hybrid\" (checked CPU \
                 attention + GPU experts), \"auto\"/\"cpu\", or — development only — set \
                 real_transformer.allow_nonfinite_attention_fallback = true"
            ),
            BackendResolutionError::HybridRequiresStrictAttention => write!(
                f,
                "compute_offload = \"hybrid\" pins attention to the checked CPU path and is \
                 incompatible with real_transformer.allow_nonfinite_attention_fallback = true; \
                 use compute_offload = \"gpu\" for GPU-attention legacy-fallback execution"
            ),
        }
    }
}

impl std::error::Error for BackendResolutionError {}

/// Error returned when an explicit GPU request cannot be honored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplicitGpuUnavailable {
    pub detail: String,
}

impl std::fmt::Display for ExplicitGpuUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "compute_offload = \"gpu\" was requested but GPU initialization failed: {}; \
             set compute_offload = \"auto\" to allow CPU fallback, or \"cpu\" to run on CPU",
            self.detail
        )
    }
}

impl std::error::Error for ExplicitGpuUnavailable {}

/// Reconcile a requested compute backend with the observed GPU
/// initialization result (Finding 5).
///
/// * `Cpu` — resolves to CPU without attempting GPU; never a fallback.
/// * `Gpu` — explicit request: GPU success resolves to GPU; GPU failure is
///   a hard error (fail closed) so the operator is never silently downgraded.
/// * `Auto` — best effort: GPU success resolves to GPU; GPU failure resolves
///   to CPU and marks `fallback_occurred`.
///
/// `gpu_init` is only consulted when GPU is requested (`Gpu`/`Auto`), and is
/// expressed as a closure so tests can inject success/failure without a real
/// device.
pub fn resolve_backend_selection<F>(
    requested: ComputeOffload,
    gpu_init: F,
) -> Result<BackendResolution, ExplicitGpuUnavailable>
where
    F: FnOnce() -> Result<(), String>,
{
    resolve_backend_selection_with_numerics(requested, false, gpu_init).map_err(|e| match e {
        BackendResolutionError::GpuUnavailable { detail, .. } => {
            ExplicitGpuUnavailable { detail }
        }
        other => ExplicitGpuUnavailable {
            detail: other.to_string(),
        },
    })
}

/// Strict-numerics-aware backend resolution (hardening pass, strict GPU
/// behaviour).
///
/// `strict_attention` is `true` when the run requires validated attention
/// numerics (the production default: `allow_nonfinite_attention_fallback
/// = false`). GPU attention performs its softmax on-device where
/// non-finite rows cannot be detected, so:
///
/// * `Gpu` + strict → hard [`BackendResolutionError::StrictGpuUnsupported`]
///   at startup — never a silent CPU-attention run reported as GPU.
/// * `Auto` + strict → resolves to CPU with a recorded reason (GPU is not
///   attempted).
/// * `Hybrid` → the explicitly named checked-CPU-attention + GPU-experts
///   split; requires strict attention and a working GPU (fails closed).
/// * Non-strict behaviour for `Cpu`/`Gpu`/`Auto` matches the legacy
///   [`resolve_backend_selection`].
pub fn resolve_backend_selection_with_numerics<F>(
    requested: ComputeOffload,
    strict_attention: bool,
    gpu_init: F,
) -> Result<BackendResolution, BackendResolutionError>
where
    F: FnOnce() -> Result<(), String>,
{
    match requested {
        ComputeOffload::Cpu => Ok(BackendResolution {
            requested,
            resolved: ResolvedBackend::Cpu,
            fallback_occurred: false,
            reason: None,
        }),
        ComputeOffload::Gpu if strict_attention => {
            Err(BackendResolutionError::StrictGpuUnsupported)
        }
        ComputeOffload::Gpu => match gpu_init() {
            Ok(()) => Ok(BackendResolution {
                requested,
                resolved: ResolvedBackend::Gpu,
                fallback_occurred: false,
                reason: None,
            }),
            Err(detail) => Err(BackendResolutionError::GpuUnavailable { requested, detail }),
        },
        ComputeOffload::Auto if strict_attention => Ok(BackendResolution {
            requested,
            resolved: ResolvedBackend::Cpu,
            fallback_occurred: false,
            reason: Some(
                "strict attention numerics: GPU on-device softmax cannot be validated, so \
                 compute_offload = \"auto\" resolved to CPU (use \"hybrid\" for checked CPU \
                 attention with GPU expert offload)"
                    .to_string(),
            ),
        }),
        ComputeOffload::Auto => match gpu_init() {
            Ok(()) => Ok(BackendResolution {
                requested,
                resolved: ResolvedBackend::Gpu,
                fallback_occurred: false,
                reason: None,
            }),
            Err(detail) => Ok(BackendResolution {
                requested,
                resolved: ResolvedBackend::Cpu,
                fallback_occurred: true,
                reason: Some(format!("GPU initialization failed: {detail}")),
            }),
        },
        ComputeOffload::Hybrid if !strict_attention => {
            Err(BackendResolutionError::HybridRequiresStrictAttention)
        }
        ComputeOffload::Hybrid => match gpu_init() {
            Ok(()) => Ok(BackendResolution {
                requested,
                resolved: ResolvedBackend::HybridCpuAttentionGpuExperts,
                fallback_occurred: false,
                reason: Some(
                    "hybrid: attention pinned to the checked CPU path; routed-expert FFN \
                     compute offloaded to the GPU"
                        .to_string(),
                ),
            }),
            Err(detail) => Err(BackendResolutionError::GpuUnavailable { requested, detail }),
        },
    }
}

fn gpu_initialization_may_run(requested: ComputeOffload, strict_attention: bool) -> bool {
    matches!(
        (requested, strict_attention),
        (ComputeOffload::Gpu, false)
            | (ComputeOffload::Auto, false)
            | (ComputeOffload::Hybrid, true)
    )
}

/// Resolve the execution plan and construct its sole production GPU backend,
/// if one is selected.
pub fn resolve_execution_context(
    requested: ComputeOffload,
    strict_attention: bool,
    geometry: GpuBackendGeometry,
    routed_expert_gpu_spec: RoutedExpertGpuSpec,
    gpu_expert_cache: Arc<crate::expert_cache::GpuExpertCache>,
) -> std::result::Result<Arc<ExecutionContext>, BackendResolutionError> {
    let context_gpu_expert_cache = gpu_expert_cache.clone();
    resolve_execution_context_with_resource_plan(
        requested,
        strict_attention,
        geometry,
        routed_expert_gpu_spec,
        context_gpu_expert_cache,
        move |geometry, resource_plan| {
            pollster::block_on(GpuBackend::try_new(
                *resource_plan,
                *geometry,
                gpu_expert_cache,
            ))
            .map(|gpu| {
                let min_storage_buffer_offset_alignment =
                    gpu.min_storage_buffer_offset_alignment();
                (
                    Arc::new(BackendBox::Gpu(gpu)),
                    min_storage_buffer_offset_alignment,
                )
            })
            .map_err(|e| e.to_string())
        },
    )
}

/// Injectable resolution boundary used by hardware-independent tests.
/// Production callers use [`resolve_execution_context`]. The factory is
/// consulted at most once and never for a CPU-only resolution.
#[cfg(test)]
pub(crate) fn resolve_execution_context_with<F>(
    requested: ComputeOffload,
    strict_attention: bool,
    geometry: GpuBackendGeometry,
    routed_expert_gpu_spec: RoutedExpertGpuSpec,
    gpu_expert_cache: Arc<crate::expert_cache::GpuExpertCache>,
    gpu_init: F,
) -> std::result::Result<Arc<ExecutionContext>, BackendResolutionError>
where
    F: FnOnce(&GpuBackendGeometry) -> std::result::Result<Arc<BackendBox>, String>,
{
    resolve_execution_context_with_device_limits(
        requested,
        strict_attention,
        geometry,
        routed_expert_gpu_spec,
        gpu_expert_cache,
        move |geometry| gpu_init(geometry).map(|backend| (backend, 1)),
    )
}

/// Intentionally unchecked context for the one engine invariant regression
/// proving that a strict GPU plan cannot silently execute through CPU.
#[cfg(test)]
pub(crate) fn test_gpu_execution_context_unchecked(
    routed_expert_backend: Arc<BackendBox>,
    routed_expert_gpu_spec: RoutedExpertGpuSpec,
) -> Arc<ExecutionContext> {
    let context_id = next_execution_context_id();
    let plan = ResolvedExecutionPlan::from_resolution(
        BackendResolution {
            requested: ComputeOffload::Hybrid,
            resolved: ResolvedBackend::HybridCpuAttentionGpuExperts,
            fallback_occurred: false,
            reason: None,
        },
        context_id,
        true,
        routed_expert_gpu_spec,
        true,
    );
    Arc::new(ExecutionContext::new(
        plan,
        Some(routed_expert_backend),
        Arc::new(crate::expert_cache::GpuExpertCache::new(1024, 0.5, 16)),
    ))
}

#[cfg(test)]
fn resolve_execution_context_with_device_limits<F>(
    requested: ComputeOffload,
    strict_attention: bool,
    geometry: GpuBackendGeometry,
    routed_expert_gpu_spec: RoutedExpertGpuSpec,
    gpu_expert_cache: Arc<crate::expert_cache::GpuExpertCache>,
    gpu_init: F,
) -> std::result::Result<Arc<ExecutionContext>, BackendResolutionError>
where
    F: FnOnce(
        &GpuBackendGeometry,
    ) -> std::result::Result<(Arc<BackendBox>, u64), String>,
{
    resolve_execution_context_with_resource_plan(
        requested,
        strict_attention,
        geometry,
        routed_expert_gpu_spec,
        gpu_expert_cache,
        move |geometry, _resource_plan| gpu_init(geometry),
    )
}

/// Resource-aware core shared by production construction and focused tests.
/// The factory receives the GPU resource plan derived from the authoritative
/// component plan for successful initialization before it creates any device
/// resources. An Auto initialization failure discards that candidate and
/// publishes the ordinary all-CPU fallback plan.
fn resolve_execution_context_with_resource_plan<F>(
    requested: ComputeOffload,
    strict_attention: bool,
    geometry: GpuBackendGeometry,
    routed_expert_gpu_spec: RoutedExpertGpuSpec,
    gpu_expert_cache: Arc<crate::expert_cache::GpuExpertCache>,
    gpu_init: F,
) -> std::result::Result<Arc<ExecutionContext>, BackendResolutionError>
where
    F: FnOnce(
        &GpuBackendGeometry,
        &GpuResourcePlan,
    ) -> std::result::Result<(Arc<BackendBox>, u64), String>,
{
    let gpu_expert_cache_available = gpu_expert_cache.capacity_bytes() > 0;
    let expert_compatibility = routed_expert_gpu_compatibility(routed_expert_gpu_spec);
    if requested == ComputeOffload::Hybrid {
        expert_compatibility.map_err(|incompatibility| {
            BackendResolutionError::RoutedExpertGpuIncompatible {
                requested,
                incompatibility,
            }
        })?;
    }
    if requested == ComputeOffload::Hybrid && !gpu_expert_cache_available {
        return Err(BackendResolutionError::RoutedExpertGpuCacheRequired);
    }

    let context_id = next_execution_context_id();
    let gpu_resource_plan = if gpu_initialization_may_run(requested, strict_attention) {
        let successful_resolution = resolve_backend_selection_with_numerics(
            requested,
            strict_attention,
            || Ok(()),
        )?;
        let successful_plan = ResolvedExecutionPlan::from_resolution(
            successful_resolution,
            context_id,
            gpu_expert_cache_available,
            routed_expert_gpu_spec,
            expert_compatibility.is_ok(),
        );
        Some(GpuResourcePlan::from_execution_plan(
            &successful_plan,
            geometry,
        )?)
    } else {
        None
    };

    let mut gpu_backend = None;
    let mut min_storage_buffer_offset_alignment = None;
    let resolution = resolve_backend_selection_with_numerics(requested, strict_attention, || {
        let resource_plan = gpu_resource_plan
            .as_ref()
            .ok_or_else(|| "GPU initialization requires a resource plan".to_string())?;
        match gpu_init(&geometry, resource_plan) {
            Ok((backend, required_alignment)) if backend.is_gpu() => {
                gpu_backend = Some(backend);
                min_storage_buffer_offset_alignment = Some(required_alignment);
                Ok(())
            }
            Ok((_, _)) => Err("GPU backend factory returned a CPU backend".to_string()),
            Err(detail) => Err(detail),
        }
    })?;

    let device_compatibility = min_storage_buffer_offset_alignment
        .map(|required_alignment| {
            routed_expert_gpu_device_compatibility(
                routed_expert_gpu_spec,
                required_alignment,
            )
        })
        .unwrap_or(Ok(()));
    if requested == ComputeOffload::Hybrid {
        device_compatibility.map_err(|incompatibility| {
            BackendResolutionError::RoutedExpertGpuIncompatible {
                requested,
                incompatibility,
            }
        })?;
    }
    let routed_expert_gpu_compatible =
        expert_compatibility.is_ok() && device_compatibility.is_ok();

    let plan = ResolvedExecutionPlan::from_resolution(
        resolution,
        context_id,
        gpu_expert_cache_available,
        routed_expert_gpu_spec,
        routed_expert_gpu_compatible,
    );
    Ok(Arc::new(ExecutionContext::new(
        plan,
        gpu_backend,
        gpu_expert_cache,
    )))
}

/// Construct an isolated CPU context. Existing CPU-only engine constructors
/// use this to preserve their behaviour without consulting process-global
/// mutable state.
pub fn cpu_execution_context() -> Arc<ExecutionContext> {
    let context_id = next_execution_context_id();
    let plan = ResolvedExecutionPlan::from_resolution(
        BackendResolution {
            requested: ComputeOffload::Cpu,
            resolved: ResolvedBackend::Cpu,
            fallback_occurred: false,
            reason: None,
        },
        context_id,
        false,
        RoutedExpertGpuSpec {
            dtype: crate::inference::WeightDtype::F32,
            d_model: 0,
            d_ff: 0,
        },
        true,
    );
    Arc::new(ExecutionContext::new(
        plan,
        None,
        Arc::new(crate::expert_cache::GpuExpertCache::new(0, 0.5, 16)),
    ))
}

// =====================================================================
// Global authoritative execution-context registry
// =====================================================================

static EXECUTION_CONTEXT: OnceLock<Arc<ExecutionContext>> = OnceLock::new();

/// Install the process-wide authoritative context. Runtime owners should clone
/// and inject this exact `Arc`; they must not reconstruct a backend from config.
pub fn set_execution_context(context: Arc<ExecutionContext>) -> Result<(), &'static str> {
    EXECUTION_CONTEXT
        .set(context)
        .map_err(|_| "execution context already installed; resolve before any token is generated")
}

/// Install a default CPU context if no context has been resolved yet.
pub fn install_default() {
    let _ = EXECUTION_CONTEXT.set(cpu_execution_context());
}

/// Active execution context. The first unresolved read permanently installs
/// the CPU default, so a caller cannot retain context A while startup later
/// replaces the registry with context B.
pub fn current_execution_context() -> Arc<ExecutionContext> {
    EXECUTION_CONTEXT.get_or_init(cpu_execution_context).clone()
}

/// Compatibility accessor for CPU-only utilities that only need a backend.
/// Model and engine runtime paths consume [`ExecutionContext`] directly.
pub fn current() -> Arc<BackendBox> {
    current_execution_context().primary_backend().clone()
}

// =====================================================================
// Unit Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_callback_reports_timeout_without_a_blocking_wait() {
        let (_tx, rx) = std::sync::mpsc::channel::<std::result::Result<(), &'static str>>();
        let mut polls = 0usize;
        let error = wait_for_bounded_callback(
            &rx,
            Duration::ZERO,
            || polls += 1,
            || None,
        )
        .unwrap_err();
        assert_eq!(error, BoundedCallbackError::Timeout);
        assert_eq!(polls, 1);
    }

    #[test]
    fn bounded_callback_distinguishes_channel_map_and_device_loss() {
        let (tx, rx) = std::sync::mpsc::channel::<std::result::Result<(), &'static str>>();
        drop(tx);
        assert_eq!(
            wait_for_bounded_callback(&rx, Duration::from_secs(1), || {}, || None)
                .unwrap_err(),
            BoundedCallbackError::ChannelDisconnected
        );

        let (tx, rx) =
            std::sync::mpsc::channel::<std::result::Result<(), &'static str>>();
        tx.send(Err("map failed")).unwrap();
        assert_eq!(
            wait_for_bounded_callback(&rx, Duration::from_secs(1), || {}, || None)
                .unwrap_err(),
            BoundedCallbackError::Callback("map failed")
        );

        let (_tx, rx) = std::sync::mpsc::channel::<std::result::Result<(), &'static str>>();
        assert_eq!(
            wait_for_bounded_callback(
                &rx,
                Duration::from_secs(1),
                || {},
                || Some("adapter reset".to_string()),
            )
            .unwrap_err(),
            BoundedCallbackError::DeviceLost("adapter reset".to_string())
        );

        assert_eq!(
            q4_parity_readback_error::<&str>(
                BoundedCallbackError::Timeout,
                Duration::from_secs(3),
            )
            .kind,
            Q4ParityGpuErrorKind::ReadbackTimeout
        );
        assert_eq!(
            q4_parity_readback_error::<&str>(
                BoundedCallbackError::ChannelDisconnected,
                Duration::from_secs(3),
            )
            .kind,
            Q4ParityGpuErrorKind::ReadbackChannel
        );
        assert_eq!(
            q4_parity_readback_error(
                BoundedCallbackError::Callback("map failed"),
                Duration::from_secs(3),
            )
            .kind,
            Q4ParityGpuErrorKind::ReadbackMap
        );
        assert_eq!(
            q4_parity_readback_error::<&str>(
                BoundedCallbackError::DeviceLost("adapter reset".to_string()),
                Duration::from_secs(3),
            )
            .kind,
            Q4ParityGpuErrorKind::DeviceLost
        );
    }

    #[test]
    fn routed_expert_io_snapshot_reads_each_monotonic_counter() {
        let counters = GpuExpertIoCounters::default();
        counters.expert_weight_uploads.fetch_add(1, Ordering::Relaxed);
        counters
            .expert_weight_upload_bytes
            .fetch_add(11, Ordering::Relaxed);
        counters.hidden_state_uploads.fetch_add(2, Ordering::Relaxed);
        counters
            .hidden_state_upload_bytes
            .fetch_add(22, Ordering::Relaxed);
        counters.queue_submissions.fetch_add(3, Ordering::Relaxed);
        counters.map_requests.fetch_add(4, Ordering::Relaxed);
        counters
            .readback_completions
            .fetch_add(5, Ordering::Relaxed);
        counters.readback_bytes.fetch_add(55, Ordering::Relaxed);

        assert_eq!(
            counters.snapshot(),
            GpuExpertIoSnapshot {
                expert_weight_uploads: 1,
                expert_weight_upload_bytes: 11,
                hidden_state_uploads: 2,
                hidden_state_upload_bytes: 22,
                queue_submissions: 3,
                map_requests: 4,
                readback_completions: 5,
                readback_bytes: 55,
            }
        );
    }

    struct TestPhysicalEntry {
        key: PhysicalExpertKey,
        bytes: u64,
        _allocation: ExpertAllocationLease,
    }

    impl PhysicalRegistryEntry for TestPhysicalEntry {
        fn physical_key(&self) -> PhysicalExpertKey {
            self.key
        }

        fn device_bytes(&self) -> u64 {
            self.bytes
        }
    }

    fn test_physical_registry(
        capacity_bytes: u64,
        workspace_bytes: u64,
    ) -> Arc<PhysicalGpuExpertRegistry<TestPhysicalEntry>> {
        PhysicalGpuExpertRegistry::new(GpuMemoryLedger::new(
            workspace_bytes,
            capacity_bytes,
        ))
    }

    fn install_test_physical_entry(
        registry: &Arc<PhysicalGpuExpertRegistry<TestPhysicalEntry>>,
        key: PhysicalExpertKey,
        bytes: u64,
    ) -> std::result::Result<Arc<TestPhysicalEntry>, String> {
        match registry
            .acquire_or_reserve(key, bytes)
            .map_err(|e| e.to_string())?
        {
            PhysicalRegistryAcquire::Hit(entry) => Ok(entry),
            PhysicalRegistryAcquire::Install(mut permit) => {
                let allocation = permit.charge_allocation()?;
                permit.install(Arc::new(TestPhysicalEntry {
                    key,
                    bytes,
                    _allocation: allocation,
                }))
            }
            PhysicalRegistryAcquire::StaleRequester => {
                Err("physical install requester generation is stale".to_string())
            }
        }
    }

    fn physical_key(expert_id: u32, generation: u64) -> PhysicalExpertKey {
        PhysicalExpertKey {
            expert_id,
            generation,
        }
    }

    #[test]
    fn physical_registry_matching_generation_hits_without_reinstall() {
        let registry = test_physical_registry(128, 0);
        let key = physical_key(7, 1);
        let first = install_test_physical_entry(&registry, key, 32).unwrap();
        let second = install_test_physical_entry(&registry, key, 32).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        let snapshot = registry.snapshot(32);
        assert_eq!(snapshot.physical_installs, 1);
        assert_eq!(snapshot.physical_entries, 1);
        assert_eq!(snapshot.expert_registry_bytes, 32);
    }

    #[test]
    fn logical_lru_to_anchor_move_reuses_matching_physical_generation() {
        let cache = crate::expert_cache::GpuExpertCache::new(128, 0.5, 3);
        let resident = Arc::new(crate::expert_cache::GpuResident::new(7, vec![0u8; 32]));
        assert_eq!(cache.demand_admit_lru(resident.clone()), Ok(true));
        let generation = cache.current_generation(7).expect("LRU generation");
        let key = physical_key(7, generation);
        let registry = test_physical_registry(128, 0);
        let physical = install_test_physical_entry(&registry, key, 32).unwrap();

        assert!(cache.claim_promotion(7, 3));
        assert_eq!(
            cache.promote_hot_existing(7),
            crate::expert_cache::GpuHotPromotionOutcome::MovedLruToAnchor
        );
        assert_eq!(cache.current_generation(7), Some(generation));
        assert!(Arc::ptr_eq(
            cache.current_admission(7).unwrap().resident(),
            &resident
        ));

        let reused = match registry.lookup_current(key) {
            PhysicalRegistryLookup::Hit(entry) => entry,
            _ => panic!("matching physical generation must remain reusable"),
        };
        assert!(Arc::ptr_eq(&physical, &reused));
        let snapshot = registry.snapshot(cache.used_bytes());
        assert_eq!(snapshot.physical_installs, 1);
        assert_eq!(snapshot.physical_entries, 1);
        assert_eq!(snapshot.expert_registry_bytes, 32);
        assert_eq!(snapshot.stale_retirements, 0);
    }

    #[test]
    fn physical_registry_retires_stale_generation_before_reinstall() {
        let registry = test_physical_registry(128, 0);
        let g1 = physical_key(7, 1);
        let old = install_test_physical_entry(&registry, g1, 32).unwrap();
        let g2 = physical_key(7, 2);
        let current = install_test_physical_entry(&registry, g2, 32).unwrap();
        assert!(!registry.contains_key(g1));
        assert!(registry.contains_key(g2));
        let after_reinstall = registry.snapshot(32);
        assert_eq!(after_reinstall.physical_installs, 2);
        assert_eq!(after_reinstall.stale_retirements, 1);
        assert_eq!(after_reinstall.expert_registry_bytes, 32);
        assert_eq!(after_reinstall.expert_live_bytes, 64);

        registry.retire_key_if_present(g1);
        assert_eq!(
            registry.snapshot(32),
            after_reinstall,
            "retiring the old key must be a no-op against stored G2"
        );
        assert!(registry.contains_key(g2));
        drop(old);
        assert_eq!(registry.snapshot(32).expert_live_bytes, 32);
        drop(current);
    }

    #[test]
    fn physical_registry_stale_requester_cannot_retire_newer_generation() {
        let registry = test_physical_registry(128, 0);
        let g1 = physical_key(7, 1);
        let g2 = physical_key(7, 2);
        let current = install_test_physical_entry(&registry, g2, 32).unwrap();
        let before = registry.snapshot(32);

        assert!(matches!(
            registry.lookup_current(g1),
            PhysicalRegistryLookup::StaleRequester
        ));
        assert_eq!(registry.snapshot(32), before);
        assert!(registry.contains_key(g2));
        assert!(!registry.contains_key(g1));
        assert_eq!(current.physical_key(), g2);
    }

    #[test]
    fn physical_registry_acquire_rejects_request_staled_after_fast_miss() {
        let registry = test_physical_registry(128, 0);
        let g1 = physical_key(7, 1);
        let g2 = physical_key(7, 2);
        assert!(matches!(
            registry.lookup_current(g1),
            PhysicalRegistryLookup::Miss
        ));

        let current = install_test_physical_entry(&registry, g2, 32).unwrap();
        let before = registry.snapshot(32);
        assert!(matches!(
            registry.acquire_or_reserve(g1, 32).unwrap(),
            PhysicalRegistryAcquire::StaleRequester
        ));
        assert_eq!(registry.snapshot(32), before);
        assert!(registry.contains_key(g2));
        assert!(!registry.contains_key(g1));
        assert_eq!(current.physical_key(), g2);
    }

    #[test]
    fn logical_miss_removes_physical_addressability_but_not_inflight_charge() {
        let registry = test_physical_registry(64, 0);
        let key = physical_key(9, 1);
        let in_flight = install_test_physical_entry(&registry, key, 40).unwrap();
        registry.retire_logical_miss(9);
        let snapshot = registry.snapshot(0);
        assert!(!registry.contains_key(key));
        assert_eq!(snapshot.physical_entries, 0);
        assert_eq!(snapshot.expert_registry_bytes, 0);
        assert_eq!(snapshot.expert_live_bytes, 40);
        drop(in_flight);
        assert_eq!(registry.snapshot(0).expert_live_bytes, 0);
    }

    #[test]
    fn physical_capacity_evicts_lru_and_never_exceeds_cap() {
        let registry = test_physical_registry(60, 0);
        drop(install_test_physical_entry(&registry, physical_key(1, 1), 30).unwrap());
        drop(install_test_physical_entry(&registry, physical_key(2, 1), 30).unwrap());
        let exact = registry.snapshot(60);
        assert_eq!(exact.expert_registry_bytes, 60);
        assert_eq!(exact.physical_entries, 2);

        drop(install_test_physical_entry(&registry, physical_key(3, 1), 30).unwrap());
        let snapshot = registry.snapshot(60);
        assert!(!registry.contains_key(physical_key(1, 1)));
        assert!(registry.contains_key(physical_key(2, 1)));
        assert!(registry.contains_key(physical_key(3, 1)));
        assert_eq!(snapshot.expert_registry_bytes, 60);
        assert!(snapshot.expert_registry_bytes <= snapshot.expert_capacity_bytes);
        assert_eq!(snapshot.physical_evictions, 1);
    }

    #[test]
    fn matching_physical_lookup_updates_registry_lru_recency() {
        let registry = test_physical_registry(60, 0);
        let g1 = physical_key(1, 1);
        let g2 = physical_key(2, 1);
        let g3 = physical_key(3, 1);
        drop(install_test_physical_entry(&registry, g1, 30).unwrap());
        drop(install_test_physical_entry(&registry, g2, 30).unwrap());

        let hit = match registry.lookup_current(g1) {
            PhysicalRegistryLookup::Hit(hit) => hit,
            PhysicalRegistryLookup::Miss => panic!("expected physical hit"),
            PhysicalRegistryLookup::StaleRequester => panic!("unexpected stale requester"),
        };
        drop(hit);
        drop(install_test_physical_entry(&registry, g3, 30).unwrap());

        assert!(registry.contains_key(g1), "matched lookup must make G1 MRU");
        assert!(!registry.contains_key(g2), "G2 must remain the LRU victim");
        assert!(registry.contains_key(g3));
    }

    #[test]
    fn oversized_physical_expert_fails_without_any_charge() {
        let registry = test_physical_registry(50, 0);
        let err = registry
            .acquire_or_reserve(physical_key(1, 1), 51)
            .err()
            .expect("oversized expert must fail");
        assert_eq!(err.requested_bytes, 51);
        assert_eq!(err.expert_capacity_bytes, 50);
        let snapshot = registry.snapshot(51);
        assert_eq!(snapshot.physical_entries, 0);
        assert_eq!(snapshot.expert_registry_bytes, 0);
        assert_eq!(snapshot.expert_live_bytes, 0);
    }

    #[test]
    fn inflight_eviction_keeps_live_charge_and_blocks_overcommit() {
        let registry = test_physical_registry(60, 0);
        let in_flight =
            install_test_physical_entry(&registry, physical_key(1, 1), 40).unwrap();
        let err = registry
            .acquire_or_reserve(physical_key(2, 1), 40)
            .err()
            .expect("in-flight live bytes must prevent overcommit");
        assert_eq!(err.expert_live_bytes, 40);
        let evicted = registry.snapshot(40);
        assert_eq!(evicted.physical_entries, 0);
        assert_eq!(evicted.expert_registry_bytes, 0);
        assert_eq!(evicted.expert_live_bytes, 40);
        drop(in_flight);
        assert_eq!(registry.snapshot(0).expert_live_bytes, 0);
        drop(install_test_physical_entry(&registry, physical_key(2, 1), 40).unwrap());
    }

    #[test]
    fn eviction_and_new_generation_perform_real_second_install() {
        let registry = test_physical_registry(64, 0);
        let first = install_test_physical_entry(&registry, physical_key(4, 1), 32).unwrap();
        registry.retire_logical_miss(4);
        drop(first);
        drop(install_test_physical_entry(&registry, physical_key(4, 2), 32).unwrap());
        let snapshot = registry.snapshot(32);
        assert_eq!(snapshot.physical_installs, 2);
        assert_eq!(snapshot.physical_entries, 1);
        assert!(registry.contains_key(physical_key(4, 2)));
    }

    #[test]
    fn failed_construction_or_charged_loser_cannot_leak_bytes() {
        let registry = test_physical_registry(64, 0);
        let key = physical_key(5, 1);
        let permit = match registry.acquire_or_reserve(key, 32).unwrap() {
            PhysicalRegistryAcquire::Install(permit) => permit,
            PhysicalRegistryAcquire::Hit(_) => panic!("unexpected physical hit"),
            PhysicalRegistryAcquire::StaleRequester => panic!("unexpected stale requester"),
        };
        drop(permit);
        assert_eq!(registry.snapshot(0).expert_live_bytes, 0);

        let mut permit = match registry.acquire_or_reserve(key, 32).unwrap() {
            PhysicalRegistryAcquire::Install(permit) => permit,
            PhysicalRegistryAcquire::Hit(_) => panic!("unexpected physical hit"),
            PhysicalRegistryAcquire::StaleRequester => panic!("unexpected stale requester"),
        };
        let allocation = permit.charge_allocation().unwrap();
        drop(permit);
        assert_eq!(registry.snapshot(0).expert_live_bytes, 32);
        drop(allocation);
        let snapshot = registry.snapshot(0);
        assert_eq!(snapshot.expert_live_bytes, 0);
        assert_eq!(snapshot.expert_registry_bytes, 0);
        assert_eq!(snapshot.physical_installs, 0);
    }

    #[test]
    fn concurrent_same_expert_demand_installs_once_and_accounts_once() {
        let registry = test_physical_registry(64, 0);
        let key = physical_key(6, 1);
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut handles = Vec::new();
        for _ in 0..2 {
            let registry = registry.clone();
            let barrier = barrier.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                install_test_physical_entry(&registry, key, 32).unwrap()
            }));
        }
        barrier.wait();
        let a = handles.remove(0).join().unwrap();
        let b = handles.remove(0).join().unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        let snapshot = registry.snapshot(32);
        assert_eq!(snapshot.physical_installs, 1);
        assert_eq!(snapshot.physical_entries, 1);
        assert_eq!(snapshot.expert_registry_bytes, 32);
        assert_eq!(snapshot.expert_live_bytes, 32);
    }

    #[test]
    fn workspace_and_snapshot_accounting_are_exact_and_exclude_host_scratch() {
        let buffer_bytes = (MAX_EXPERT_D_FF * std::mem::size_of::<f32>()) as u64;
        let workspace_bytes = expert_workspace_device_bytes(buffer_bytes, EXPERT_WORKSPACE_POOL)
            .expect("test workspace bytes");
        assert_eq!(
            workspace_bytes,
            buffer_bytes * EXPERT_WORKSPACE_DEVICE_BUFFERS * EXPERT_WORKSPACE_POOL as u64
        );
        // `Vec<f32>` scratch has the same logical length as one device buffer,
        // but is intentionally absent from the five-buffer formula above.
        let host_scratch_bytes = buffer_bytes * EXPERT_WORKSPACE_POOL as u64;
        assert_ne!(workspace_bytes, workspace_bytes + host_scratch_bytes);

        let registry = test_physical_registry(64, workspace_bytes);
        drop(install_test_physical_entry(&registry, physical_key(1, 1), 32).unwrap());
        let snapshot = registry.snapshot(32);
        assert_eq!(snapshot.workspace_bytes, workspace_bytes);
        assert_eq!(snapshot.total_tracked_bytes, 32 + workspace_bytes);
        assert_eq!(snapshot.total_tracked_bytes, snapshot.expert_live_bytes + snapshot.workspace_bytes);
        assert!(snapshot.expert_registry_bytes <= snapshot.expert_live_bytes);
        assert!(snapshot.expert_registry_bytes <= snapshot.expert_capacity_bytes);
    }

    #[test]
    fn kv_offset_symmetric_matches_shared_stride() {
        // Symmetric case: k_dim == v_dim. Layout is a single contiguous
        // [K|V] block per layer; every V slot sits exactly one K region
        // (max_seq_len * k_dim) past the same-position K slot.
        let (max_seq, k_dim, v_dim) = (8usize, 4usize, 4usize);
        assert_eq!(kv_layer_stride_elems(max_seq, k_dim, v_dim), 8 * 8);
        // layer 0, K, pos 0
        assert_eq!(kv_offset_elems(0, 0, 0, max_seq, k_dim, v_dim), 0);
        // layer 0, K, pos 3
        assert_eq!(kv_offset_elems(0, 0, 3, max_seq, k_dim, v_dim), 3 * 4);
        // layer 0, V, pos 3 = K region (8*4) + 3*4
        assert_eq!(
            kv_offset_elems(0, 1, 3, max_seq, k_dim, v_dim),
            8 * 4 + 3 * 4
        );
        // layer 1 starts after one full [K|V] stride
        assert_eq!(
            kv_offset_elems(1, 0, 0, max_seq, k_dim, v_dim),
            8 * (4 + 4)
        );
    }

    #[test]
    fn kv_offset_asymmetric_v_uses_independent_strides() {
        // Finding 12: v_dim != k_dim. K positions must stride by k_dim,
        // V positions by v_dim, and the V region base must skip the full
        // K region (max_seq_len * k_dim), not max_seq_len * v_dim.
        let (max_seq, k_dim, v_dim) = (16usize, 8usize, 12usize);
        assert_eq!(kv_layer_stride_elems(max_seq, k_dim, v_dim), 16 * (8 + 12));
        // K strides by k_dim
        assert_eq!(kv_offset_elems(0, 0, 2, max_seq, k_dim, v_dim), 2 * 8);
        // V region begins at max_seq_len * k_dim (NOT * v_dim)
        assert_eq!(kv_offset_elems(0, 1, 0, max_seq, k_dim, v_dim), 16 * 8);
        // V strides by v_dim
        assert_eq!(
            kv_offset_elems(0, 1, 2, max_seq, k_dim, v_dim),
            16 * 8 + 2 * 12
        );
        // layer 2 offset = 2 * (max_seq * (k_dim + v_dim))
        assert_eq!(
            kv_offset_elems(2, 0, 0, max_seq, k_dim, v_dim),
            2 * 16 * (8 + 12)
        );
        // V slot in layer 2 also correct
        assert_eq!(
            kv_offset_elems(2, 1, 1, max_seq, k_dim, v_dim),
            2 * 16 * (8 + 12) + 16 * 8 + 1 * 12
        );
    }

    #[test]
    fn kv_offset_bytes_are_elems_times_four() {
        // The byte-offset accessor must scale the element index by 4 and
        // never overflow for a realistically deep asymmetric model.
        let idx = kv_offset_elems(40, 1, 4000, 4096, 1024, 1536);
        assert_eq!((idx as u64) * 4, kv_offset_bytes_reference(idx));
    }

    fn kv_offset_bytes_reference(idx: usize) -> u64 {
        (idx as u64) * 4
    }

    fn adapter(
        name: &str,
        backend: wgpu::Backend,
        device_type: wgpu::DeviceType,
    ) -> AdapterMetadata {
        AdapterMetadata {
            name: name.to_string(),
            vendor: 0x10de,
            device: 0x27b8,
            device_type,
            driver: "test-driver".to_string(),
            driver_info: "test-driver-info".to_string(),
            backend,
        }
    }

    fn test_gpu_geometry() -> GpuBackendGeometry {
        GpuBackendGeometry {
            num_layers: 2,
            max_seq_len: 128,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            v_head_dim: 8,
            q4_truncation_tolerance: 0,
        }
    }

    fn test_routed_expert_gpu_spec(dtype: crate::inference::WeightDtype) -> RoutedExpertGpuSpec {
        test_routed_expert_gpu_spec_with_shape(dtype, 32, 64)
    }

    fn test_routed_expert_gpu_spec_with_shape(
        dtype: crate::inference::WeightDtype,
        d_model: usize,
        d_ff: usize,
    ) -> RoutedExpertGpuSpec {
        RoutedExpertGpuSpec {
            dtype,
            d_model,
            d_ff,
        }
    }

    fn test_gpu_backend() -> Arc<BackendBox> {
        Arc::new(BackendBox::TestGpu(TestGpuBackend::success(1.0)))
    }

    fn test_gpu_expert_cache(capacity_bytes: usize) -> Arc<crate::expert_cache::GpuExpertCache> {
        Arc::new(crate::expert_cache::GpuExpertCache::new(
            capacity_bytes,
            0.5,
            16,
        ))
    }

    fn test_resolved_plan(
        resolved: ResolvedBackend,
        dtype: crate::inference::WeightDtype,
    ) -> ResolvedExecutionPlan {
        let requested = match resolved {
            ResolvedBackend::Cpu => ComputeOffload::Cpu,
            ResolvedBackend::Gpu => ComputeOffload::Gpu,
            ResolvedBackend::HybridCpuAttentionGpuExperts => ComputeOffload::Hybrid,
        };
        ResolvedExecutionPlan::from_resolution(
            BackendResolution {
                requested,
                resolved,
                fallback_occurred: false,
                reason: None,
            },
            next_execution_context_id(),
            true,
            test_routed_expert_gpu_spec(dtype),
            true,
        )
    }

    fn qwen3_coder_geometry() -> GpuBackendGeometry {
        GpuBackendGeometry {
            num_layers: 48,
            max_seq_len: 4096,
            num_heads: 32,
            num_kv_heads: 4,
            head_dim: 128,
            v_head_dim: 128,
            q4_truncation_tolerance: 0,
        }
    }

    #[test]
    fn hybrid_q4_resource_plan_is_routed_expert_only_with_zero_kv() {
        let plan = test_resolved_plan(
            ResolvedBackend::HybridCpuAttentionGpuExperts,
            crate::inference::WeightDtype::Q4_0,
        );
        let resources = GpuResourcePlan::from_execution_plan(&plan, qwen3_coder_geometry())
            .expect("strict Hybrid Q4 resource plan");

        assert_eq!(resources.mode(), GpuBackendMode::RoutedExpertsOnly);
        assert_eq!(resources.kv_allocation_bytes(), 0);
        assert_eq!(resources.dense_allocation_bytes(), 0);
        assert!(!resources.constructs_dense_resources());
        assert!(!resources.constructs_attention_resources());
        assert!(!resources.constructs_f32_matmul_pipeline());
        assert!(resources.constructs_q4_0_matmul_pipeline());
        assert!(resources.has_capability(GpuCapability::RoutedExperts));
        assert_eq!(resources.expert_workspace_allocation_bytes(), 1_310_720);
    }

    #[test]
    fn hybrid_disabled_component_guards_return_typed_errors() {
        let plan = test_resolved_plan(
            ResolvedBackend::HybridCpuAttentionGpuExperts,
            crate::inference::WeightDtype::Q4_0,
        );
        let resources = GpuResourcePlan::from_execution_plan(&plan, qwen3_coder_geometry())
            .expect("strict Hybrid Q4 resource plan");

        for capability in [
            GpuCapability::Dense,
            GpuCapability::Attention,
            GpuCapability::Kv,
        ] {
            assert_eq!(
                resources.require_capability(capability),
                Err(GpuCapabilityUnavailable {
                    mode: GpuBackendMode::RoutedExpertsOnly,
                    capability,
                })
            );
        }
        assert_eq!(resources.require_capability(GpuCapability::RoutedExperts), Ok(()));
    }

    #[test]
    fn hybrid_component_factory_skips_dense_attention_and_kv_construction() {
        let plan = test_resolved_plan(
            ResolvedBackend::HybridCpuAttentionGpuExperts,
            crate::inference::WeightDtype::Q4_0,
        );
        let plan = GpuResourcePlan::from_execution_plan(&plan, qwen3_coder_geometry())
            .expect("strict Hybrid Q4 resource plan");
        let dense_called = std::cell::Cell::new(false);
        let attention_called = std::cell::Cell::new(false);

        let resources = GpuComponentResources::<(), ()>::try_new(
            plan,
            || {
                dense_called.set(true);
                Ok(())
            },
            |_| {
                attention_called.set(true);
                Ok(())
            },
        )
        .expect("Hybrid component resource construction");

        assert!(!dense_called.get());
        assert!(!attention_called.get());
        for capability in [GpuCapability::Dense, GpuCapability::Attention] {
            let err = resources
                .dense(capability)
                .expect_err("disabled production resource accessor must fail");
            assert_eq!(
                err.downcast_ref::<GpuCapabilityUnavailable>(),
                Some(&GpuCapabilityUnavailable {
                    mode: GpuBackendMode::RoutedExpertsOnly,
                    capability,
                })
            );
        }
        let err = resources
            .attention()
            .expect_err("disabled production attention accessor must fail");
        assert_eq!(
            err.downcast_ref::<GpuCapabilityUnavailable>(),
            Some(&GpuCapabilityUnavailable {
                mode: GpuBackendMode::RoutedExpertsOnly,
                capability: GpuCapability::Attention,
            })
        );
        let err = resources
            .require_capability(GpuCapability::Kv)
            .expect_err("disabled production KV guard must fail");
        assert_eq!(
            err.downcast_ref::<GpuCapabilityUnavailable>(),
            Some(&GpuCapabilityUnavailable {
                mode: GpuBackendMode::RoutedExpertsOnly,
                capability: GpuCapability::Kv,
            })
        );
    }

    #[test]
    fn full_gpu_resource_plan_retains_dense_attention_and_exact_kv_geometry() {
        let plan = test_resolved_plan(
            ResolvedBackend::Gpu,
            crate::inference::WeightDtype::Q4_0,
        );
        let geometry = qwen3_coder_geometry();
        let resources = GpuResourcePlan::from_execution_plan(&plan, geometry)
            .expect("full GPU resource plan");

        assert_eq!(resources.mode(), GpuBackendMode::Full);
        assert_eq!(resources.kv_allocation_bytes(), 805_306_368);
        assert_eq!(resources.kv_allocation_bytes(), geometry.kv_allocation_bytes().unwrap());
        assert_eq!(resources.dense_allocation_bytes(), DENSE_BUFFER_BYTES * 5);
        assert!(resources.constructs_dense_resources());
        assert!(resources.constructs_attention_resources());
        assert!(resources.constructs_f32_matmul_pipeline());
        assert!(resources.constructs_q4_0_matmul_pipeline());
        for capability in [
            GpuCapability::RoutedExperts,
            GpuCapability::Dense,
            GpuCapability::Attention,
            GpuCapability::Kv,
        ] {
            assert!(resources.has_capability(capability));
        }
    }

    #[test]
    fn full_component_factory_constructs_and_exposes_all_optional_resources() {
        let plan = test_resolved_plan(
            ResolvedBackend::Gpu,
            crate::inference::WeightDtype::Q4_0,
        );
        let plan = GpuResourcePlan::from_execution_plan(&plan, qwen3_coder_geometry())
            .expect("full GPU resource plan");
        let dense_called = std::cell::Cell::new(false);
        let attention_called = std::cell::Cell::new(false);

        let resources = GpuComponentResources::try_new(
            plan,
            || {
                dense_called.set(true);
                Ok(7u8)
            },
            |dense| {
                attention_called.set(true);
                assert_eq!(*dense, 7);
                Ok(11u8)
            },
        )
        .expect("Full component resource construction");

        assert!(dense_called.get());
        assert!(attention_called.get());
        assert_eq!(*resources.dense(GpuCapability::Dense).unwrap(), 7);
        assert_eq!(*resources.attention().unwrap(), 11);
        assert!(resources.require_capability(GpuCapability::Kv).is_ok());
    }

    #[test]
    fn oversized_startup_buffer_is_rejected_before_wgpu_allocation() {
        let limits = wgpu::Limits {
            max_buffer_size: 268_435_456,
            max_storage_buffer_binding_size: 268_435_456,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            validate_startup_buffer(
                "kv_cache",
                805_306_368,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                &limits,
            ),
            Err(GpuStartupAllocationError::ExceedsMaxBufferSize {
                label: "kv_cache".to_string(),
                requested: 805_306_368,
                maximum: 268_435_456,
            })
        );

        let limits = wgpu::Limits {
            max_buffer_size: 1_000,
            max_storage_buffer_binding_size: 100,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            validate_startup_buffer(
                "storage",
                200,
                wgpu::BufferUsages::STORAGE,
                &limits,
            ),
            Err(GpuStartupAllocationError::ExceedsMaxStorageBindingSize {
                label: "storage".to_string(),
                requested: 200,
                maximum: 100,
            })
        );
    }

    #[test]
    fn resolver_passes_hybrid_component_plan_to_gpu_factory() {
        let mut observed = None;
        let context = resolve_execution_context_with_resource_plan(
            ComputeOffload::Hybrid,
            true,
            qwen3_coder_geometry(),
            test_routed_expert_gpu_spec(crate::inference::WeightDtype::Q4_0),
            test_gpu_expert_cache(1024),
            |_, resources| {
                observed = Some(*resources);
                Ok((test_gpu_backend(), 256))
            },
        )
        .expect("strict Hybrid resolves with injected GPU");

        let resources = observed.expect("resource-aware factory was called");
        assert_eq!(resources.mode(), GpuBackendMode::RoutedExpertsOnly);
        assert_eq!(resources.kv_allocation_bytes(), 0);
        assert!(!resources.constructs_dense_resources());
        assert!(!resources.constructs_attention_resources());
        assert_eq!(context.plan().kv(), ExecutionPlane::Cpu);
        assert_eq!(context.plan().attention(), ExecutionPlane::Cpu);
        assert_eq!(context.plan().routed_experts(), ExecutionPlane::Gpu);
    }

    #[test]
    fn routed_expert_gpu_compatibility_accepts_f32() {
        assert_eq!(
            routed_expert_gpu_compatibility(test_routed_expert_gpu_spec(
                crate::inference::WeightDtype::F32,
            )),
            Ok(())
        );
    }

    #[test]
    fn routed_expert_gpu_compatibility_accepts_aligned_q4_0() {
        assert_eq!(
            routed_expert_gpu_compatibility(test_routed_expert_gpu_spec(
                crate::inference::WeightDtype::Q4_0,
            )),
            Ok(())
        );
    }

    #[test]
    fn routed_expert_gpu_compatibility_accepts_workspace_limit() {
        for dtype in [
            crate::inference::WeightDtype::F32,
            crate::inference::WeightDtype::Q4_0,
        ] {
            assert_eq!(
                routed_expert_gpu_compatibility(test_routed_expert_gpu_spec_with_shape(
                    dtype,
                    MAX_EXPERT_D_FF,
                    MAX_EXPERT_D_FF,
                )),
                Ok(())
            );
        }
    }

    #[test]
    fn routed_expert_gpu_compatibility_rejects_d_model_above_workspace() {
        let err = routed_expert_gpu_compatibility(test_routed_expert_gpu_spec_with_shape(
            crate::inference::WeightDtype::F32,
            MAX_EXPERT_D_FF + 1,
            64,
        ));
        assert_eq!(
            err,
            Err(RoutedExpertGpuIncompatibility::ShapeExceedsWorkspace {
                d_model: MAX_EXPERT_D_FF + 1,
                d_ff: 64,
                max: MAX_EXPERT_D_FF,
            })
        );
    }

    #[test]
    fn routed_expert_gpu_compatibility_rejects_d_ff_above_workspace() {
        let err = routed_expert_gpu_compatibility(test_routed_expert_gpu_spec_with_shape(
            crate::inference::WeightDtype::Q4_0,
            64,
            MAX_EXPERT_D_FF + 32,
        ));
        assert_eq!(
            err,
            Err(RoutedExpertGpuIncompatibility::ShapeExceedsWorkspace {
                d_model: 64,
                d_ff: MAX_EXPERT_D_FF + 32,
                max: MAX_EXPERT_D_FF,
            })
        );
    }

    #[test]
    fn hybrid_workspace_rejection_skips_gpu_factory() {
        for spec in [
            test_routed_expert_gpu_spec_with_shape(
                crate::inference::WeightDtype::F32,
                MAX_EXPERT_D_FF + 1,
                64,
            ),
            test_routed_expert_gpu_spec_with_shape(
                crate::inference::WeightDtype::Q4_0,
                64,
                MAX_EXPERT_D_FF + 32,
            ),
        ] {
            let mut attempts = 0;
            let err = resolve_execution_context_with(
                ComputeOffload::Hybrid,
                true,
                test_gpu_geometry(),
                spec,
                test_gpu_expert_cache(1024),
                |_| {
                    attempts += 1;
                    Ok(test_gpu_backend())
                },
            )
            .unwrap_err();
            assert_eq!(attempts, 0);
            assert!(matches!(
                err,
                BackendResolutionError::RoutedExpertGpuIncompatible {
                    requested: ComputeOffload::Hybrid,
                    incompatibility: RoutedExpertGpuIncompatibility::ShapeExceedsWorkspace { .. },
                }
            ));
        }
    }

    #[test]
    fn f32_projection_layout_accepts_device_alignment() {
        let layout = f32_expert_projection_layout(
            test_routed_expert_gpu_spec(crate::inference::WeightDtype::F32),
            256,
        )
        .unwrap();
        assert_eq!(layout.projection_bytes, 32 * 64 * 4);
        assert_eq!(layout.projection_offset, (32 * 64 * 4) as u64);
        assert_eq!(layout.required_bytes, 3 * 32 * 64 * 4);
    }

    #[test]
    fn f32_upload_plan_excludes_trailing_host_padding() {
        let layout = f32_expert_projection_layout(
            test_routed_expert_gpu_spec(crate::inference::WeightDtype::F32),
            256,
        )
        .unwrap();
        let host_bytes = layout.required_bytes + 4096;
        let plan = f32_expert_upload_plan(layout, host_bytes, 32, 64).unwrap();

        assert_eq!(plan.device_bytes, layout.required_bytes as u64);
        assert_eq!(plan.projection_offset, layout.projection_offset);
        assert!(plan.device_bytes < host_bytes as u64);
    }

    #[test]
    fn f32_projection_layout_rejects_device_misalignment() {
        let err = f32_expert_projection_layout(
            test_routed_expert_gpu_spec_with_shape(
                crate::inference::WeightDtype::F32,
                32,
                65,
            ),
            256,
        )
        .unwrap_err();
        assert_eq!(
            err,
            RoutedExpertGpuIncompatibility::F32StorageOffsetMisaligned {
                projection_bytes: 32 * 65 * 4,
                required_alignment: 256,
            }
        );
    }

    #[test]
    fn f32_projection_layout_rejects_checked_overflow() {
        let err = f32_expert_projection_layout(
            test_routed_expert_gpu_spec_with_shape(
                crate::inference::WeightDtype::F32,
                usize::MAX,
                2,
            ),
            256,
        )
        .unwrap_err();
        assert_eq!(
            err,
            RoutedExpertGpuIncompatibility::F32ProjectionSizeOverflow {
                d_model: usize::MAX,
                d_ff: 2,
            }
        );
    }

    #[test]
    fn hybrid_rejects_unsupported_expert_dtypes_before_gpu_factory() {
        for dtype in [
            crate::inference::WeightDtype::Q8_0,
            crate::inference::WeightDtype::Mixed,
            crate::inference::WeightDtype::F16,
        ] {
            let mut attempts = 0;
            let err = resolve_execution_context_with(
                ComputeOffload::Hybrid,
                true,
                test_gpu_geometry(),
                test_routed_expert_gpu_spec(dtype),
                test_gpu_expert_cache(1024),
                |_| {
                    attempts += 1;
                    Ok(test_gpu_backend())
                },
            )
            .unwrap_err();
            assert_eq!(attempts, 0, "GPU factory ran for incompatible {dtype:?}");
            assert!(matches!(
                err,
                BackendResolutionError::RoutedExpertGpuIncompatible {
                    requested: ComputeOffload::Hybrid,
                    incompatibility: RoutedExpertGpuIncompatibility::UnsupportedDtype {
                        dtype: rejected,
                    },
                } if rejected == dtype
            ));
        }
    }

    #[test]
    fn hybrid_rejects_misaligned_q4_0_before_gpu_factory() {
        let mut attempts = 0;
        let err = resolve_execution_context_with(
            ComputeOffload::Hybrid,
            true,
            test_gpu_geometry(),
            RoutedExpertGpuSpec {
                dtype: crate::inference::WeightDtype::Q4_0,
                d_model: 31,
                d_ff: 64,
            },
            test_gpu_expert_cache(1024),
            |_| {
                attempts += 1;
                Ok(test_gpu_backend())
            },
        )
        .unwrap_err();
        assert_eq!(attempts, 0);
        assert!(matches!(
            err,
            BackendResolutionError::RoutedExpertGpuIncompatible {
                requested: ComputeOffload::Hybrid,
                incompatibility: RoutedExpertGpuIncompatibility::MisalignedQ4_0 {
                    d_model: 31,
                    d_ff: 64,
                    block_elems: 32,
                },
            }
        ));
        assert!(err.to_string().contains("d_model=31"));
    }

    #[test]
    fn legacy_gpu_and_auto_plans_keep_incompatible_experts_on_cpu() {
        for requested in [ComputeOffload::Gpu, ComputeOffload::Auto] {
            let mut attempts = 0;
            let context = resolve_execution_context_with(
                requested,
                false,
                test_gpu_geometry(),
                test_routed_expert_gpu_spec(crate::inference::WeightDtype::Q8_0),
                test_gpu_expert_cache(1024),
                |_| {
                    attempts += 1;
                    Ok(test_gpu_backend())
                },
            )
            .unwrap();
            assert_eq!(attempts, 1);
            assert_eq!(context.plan().resolved(), ResolvedBackend::Gpu);
            assert_eq!(context.plan().attention(), ExecutionPlane::Gpu);
            assert_eq!(context.plan().routed_experts(), ExecutionPlane::Cpu);
            assert!(!context.routed_expert_backend().is_gpu());
        }
    }

    #[test]
    fn hybrid_device_misalignment_is_typed_after_one_gpu_initialization() {
        let mut attempts = 0;
        let err = resolve_execution_context_with_device_limits(
            ComputeOffload::Hybrid,
            true,
            test_gpu_geometry(),
            test_routed_expert_gpu_spec_with_shape(
                crate::inference::WeightDtype::F32,
                32,
                65,
            ),
            test_gpu_expert_cache(1024),
            |_| {
                attempts += 1;
                Ok((test_gpu_backend(), 256))
            },
        )
        .unwrap_err();
        assert_eq!(attempts, 1);
        assert_eq!(
            err,
            BackendResolutionError::RoutedExpertGpuIncompatible {
                requested: ComputeOffload::Hybrid,
                incompatibility:
                    RoutedExpertGpuIncompatibility::F32StorageOffsetMisaligned {
                        projection_bytes: 32 * 65 * 4,
                        required_alignment: 256,
                    },
            }
        );
    }

    #[test]
    fn legacy_gpu_and_auto_device_misalignment_keeps_experts_on_cpu() {
        for requested in [ComputeOffload::Gpu, ComputeOffload::Auto] {
            let context = resolve_execution_context_with_device_limits(
                requested,
                false,
                test_gpu_geometry(),
                test_routed_expert_gpu_spec_with_shape(
                    crate::inference::WeightDtype::F32,
                    32,
                    65,
                ),
                test_gpu_expert_cache(1024),
                |_| Ok((test_gpu_backend(), 256)),
            )
            .unwrap();
            assert_eq!(context.plan().resolved(), ResolvedBackend::Gpu);
            assert_eq!(context.plan().attention(), ExecutionPlane::Gpu);
            assert_eq!(context.plan().routed_experts(), ExecutionPlane::Cpu);
            assert!(!context.routed_expert_backend().is_gpu());
        }
    }

    #[test]
    fn cpu_execution_plan_is_all_cpu_and_skips_gpu_factory() {
        let mut attempts = 0;
        let context = resolve_execution_context_with(
            ComputeOffload::Cpu,
            true,
            test_gpu_geometry(),
            test_routed_expert_gpu_spec(crate::inference::WeightDtype::F32),
            test_gpu_expert_cache(0),
            |_| {
                attempts += 1;
                Ok(test_gpu_backend())
            },
        )
        .unwrap();

        assert_eq!(attempts, 0);
        assert_eq!(context.plan().requested(), ComputeOffload::Cpu);
        assert_eq!(context.plan().resolved(), ResolvedBackend::Cpu);
        assert!(context
            .plan()
            .component_planes()
            .iter()
            .all(|(_, plane)| *plane == ExecutionPlane::Cpu));
        assert!(!context.primary_backend().is_gpu());
        assert_eq!(context.id(), context.plan().context_id());
    }

    #[test]
    fn hybrid_plan_constructs_exactly_one_gpu_context_and_keeps_only_experts_on_gpu() {
        let mut attempts = 0;
        let expected_backend = test_gpu_backend();
        let factory_backend = expected_backend.clone();
        let expected_cache = test_gpu_expert_cache(1024);
        let context = resolve_execution_context_with_device_limits(
            ComputeOffload::Hybrid,
            true,
            test_gpu_geometry(),
            test_routed_expert_gpu_spec(crate::inference::WeightDtype::F32),
            expected_cache.clone(),
            |_| {
                attempts += 1;
                Ok((factory_backend, 256))
            },
        )
        .unwrap();

        let plan = context.plan();
        assert_eq!(attempts, 1);
        assert_eq!(plan.requested(), ComputeOffload::Hybrid);
        assert_eq!(
            plan.resolved(),
            ResolvedBackend::HybridCpuAttentionGpuExperts
        );
        assert!(!plan.fallback_occurred());
        assert_eq!(plan.embeddings(), ExecutionPlane::Cpu);
        assert_eq!(plan.lm_head(), ExecutionPlane::Cpu);
        assert_eq!(plan.dense_projections(), ExecutionPlane::Cpu);
        assert_eq!(plan.attention(), ExecutionPlane::Cpu);
        assert_eq!(plan.kv(), ExecutionPlane::Cpu);
        assert_eq!(plan.router(), ExecutionPlane::Cpu);
        assert_eq!(plan.routed_experts(), ExecutionPlane::Gpu);
        assert!(Arc::ptr_eq(
            context.routed_expert_backend(),
            &expected_backend
        ));
        assert!(Arc::ptr_eq(context.gpu_expert_cache(), &expected_cache));
        assert!(!context.attention_backend().is_gpu());
    }

    #[test]
    #[should_panic(expected = "resolved component planes and GPU backend ownership disagree")]
    fn execution_context_rejects_gpu_plane_without_gpu_backend() {
        let plan = ResolvedExecutionPlan::from_resolution(
            BackendResolution {
                requested: ComputeOffload::Gpu,
                resolved: ResolvedBackend::Gpu,
                fallback_occurred: false,
                reason: None,
            },
            next_execution_context_id(),
            true,
            test_routed_expert_gpu_spec(crate::inference::WeightDtype::F32),
            true,
        );
        let _ = ExecutionContext::new(plan, None, test_gpu_expert_cache(1024));
    }

    #[test]
    #[should_panic(expected = "GPU routed-expert plane requires non-zero expert-weight capacity")]
    fn execution_context_rejects_gpu_experts_without_cache() {
        let plan = ResolvedExecutionPlan::from_resolution(
            BackendResolution {
                requested: ComputeOffload::Hybrid,
                resolved: ResolvedBackend::HybridCpuAttentionGpuExperts,
                fallback_occurred: false,
                reason: None,
            },
            next_execution_context_id(),
            true,
            test_routed_expert_gpu_spec(crate::inference::WeightDtype::F32),
            true,
        );
        let _ = ExecutionContext::new(plan, Some(test_gpu_backend()), test_gpu_expert_cache(0));
    }

    #[test]
    fn explicit_gpu_context_resolution_fails_closed_on_initialization_failure() {
        let err = resolve_execution_context_with(
            ComputeOffload::Gpu,
            false,
            test_gpu_geometry(),
            test_routed_expert_gpu_spec(crate::inference::WeightDtype::F32),
            test_gpu_expert_cache(1024),
            |_| Err("adapter incompatible with required PUSH_CONSTANTS".to_string()),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            BackendResolutionError::GpuUnavailable {
                requested: ComputeOffload::Gpu,
                ..
            }
        ));
        assert!(err.to_string().contains("PUSH_CONSTANTS"));
    }

    #[test]
    fn auto_context_fallback_is_explicitly_cpu_and_preserves_request() {
        let context = resolve_execution_context_with(
            ComputeOffload::Auto,
            false,
            test_gpu_geometry(),
            test_routed_expert_gpu_spec(crate::inference::WeightDtype::F32),
            test_gpu_expert_cache(1024),
            |_| Err("no compatible adapter".to_string()),
        )
        .unwrap();
        let plan = context.plan();
        assert_eq!(plan.requested(), ComputeOffload::Auto);
        assert_eq!(plan.resolved(), ResolvedBackend::Cpu);
        assert!(plan.fallback_occurred());
        assert!(plan.reason().unwrap().contains("no compatible adapter"));
        assert!(plan
            .component_planes()
            .iter()
            .all(|(_, plane)| *plane == ExecutionPlane::Cpu));
        assert!(!context.primary_backend().is_gpu());
    }

    #[test]
    fn invalid_full_gpu_geometry_fails_before_factory_or_runtime() {
        let mut attempts = 0;
        let mut geometry = test_gpu_geometry();
        geometry.num_heads = 0;
        let err = resolve_execution_context_with(
            ComputeOffload::Gpu,
            false,
            geometry,
            test_routed_expert_gpu_spec(crate::inference::WeightDtype::F32),
            test_gpu_expert_cache(1024),
            |_| {
                attempts += 1;
                Ok(test_gpu_backend())
            },
        )
        .unwrap_err();
        assert_eq!(attempts, 0);
        assert!(matches!(
            err,
            BackendResolutionError::InvalidGeometry { .. }
        ));
    }

    #[test]
    fn hybrid_plan_does_not_validate_unused_attention_or_kv_geometry() {
        let mut geometry = test_gpu_geometry();
        geometry.num_layers = 0;
        geometry.max_seq_len = 0;
        geometry.num_heads = 0;
        geometry.num_kv_heads = 0;
        geometry.head_dim = 0;
        geometry.v_head_dim = 0;
        let context = resolve_execution_context_with_resource_plan(
            ComputeOffload::Hybrid,
            true,
            geometry,
            test_routed_expert_gpu_spec(crate::inference::WeightDtype::Q4_0),
            test_gpu_expert_cache(1024),
            |_, resources| {
                assert_eq!(resources.mode(), GpuBackendMode::RoutedExpertsOnly);
                assert_eq!(resources.kv_allocation_bytes(), 0);
                Ok((test_gpu_backend(), 256))
            },
        )
        .expect("unused attention/KV geometry cannot reject Hybrid expert resources");
        assert_eq!(context.plan().routed_experts(), ExecutionPlane::Gpu);
    }

    #[test]
    fn backend_factory_cannot_claim_gpu_with_a_cpu_backend() {
        let err = resolve_execution_context_with(
            ComputeOffload::Hybrid,
            true,
            test_gpu_geometry(),
            test_routed_expert_gpu_spec(crate::inference::WeightDtype::F32),
            test_gpu_expert_cache(1024),
            |_| Ok(Arc::new(BackendBox::Cpu(CandleBackend::new()))),
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("GPU backend factory returned a CPU backend"));
    }

    #[test]
    fn hybrid_requires_an_executable_gpu_expert_cache_before_initialization() {
        let mut attempts = 0;
        let err = resolve_execution_context_with(
            ComputeOffload::Hybrid,
            true,
            test_gpu_geometry(),
            test_routed_expert_gpu_spec(crate::inference::WeightDtype::F32),
            test_gpu_expert_cache(0),
            |_| {
                attempts += 1;
                Ok(test_gpu_backend())
            },
        )
        .unwrap_err();
        assert_eq!(attempts, 0);
        assert_eq!(err, BackendResolutionError::RoutedExpertGpuCacheRequired);
    }

    #[test]
    fn full_gpu_without_expert_cache_reports_experts_on_cpu() {
        let context = resolve_execution_context_with(
            ComputeOffload::Gpu,
            false,
            test_gpu_geometry(),
            test_routed_expert_gpu_spec(crate::inference::WeightDtype::F32),
            test_gpu_expert_cache(0),
            |_| Ok(test_gpu_backend()),
        )
        .unwrap();
        assert_eq!(context.plan().attention(), ExecutionPlane::Gpu);
        assert_eq!(context.plan().kv(), ExecutionPlane::Gpu);
        assert_eq!(context.plan().routed_experts(), ExecutionPlane::Cpu);
        assert!(!context.routed_expert_backend().is_gpu());
    }

    // ---- Finding 5: explicit GPU requests fail closed ----

    #[test]
    fn explicit_gpu_request_with_init_failure_errors() {
        let out = resolve_backend_selection(ComputeOffload::Gpu, || Err("no device".to_string()));
        assert!(
            out.is_err(),
            "explicit GPU request must fail closed when init fails"
        );
    }

    #[test]
    fn explicit_gpu_request_with_success_resolves_to_gpu() {
        let out =
            resolve_backend_selection(ComputeOffload::Gpu, || Ok(())).expect("gpu init succeeded");
        assert_eq!(out.resolved, ResolvedBackend::Gpu);
        assert!(!out.fallback_occurred);
    }

    #[test]
    fn auto_selection_with_init_failure_falls_back_to_cpu_and_marks_it() {
        let out = resolve_backend_selection(ComputeOffload::Auto, || Err("no device".to_string()))
            .expect("auto must not fail closed");
        assert_eq!(out.resolved, ResolvedBackend::Cpu);
        assert!(
            out.fallback_occurred,
            "auto GPU->CPU demotion must be recorded as a fallback"
        );
    }

    #[test]
    fn auto_selection_with_success_resolves_to_gpu_without_fallback() {
        let out = resolve_backend_selection(ComputeOffload::Auto, || Ok(())).unwrap();
        assert_eq!(out.resolved, ResolvedBackend::Gpu);
        assert!(!out.fallback_occurred);
    }

    #[test]
    fn explicit_cpu_resolves_to_cpu_without_attempting_gpu() {
        let mut attempted = false;
        let out = resolve_backend_selection(ComputeOffload::Cpu, || {
            attempted = true;
            Ok(())
        })
        .unwrap();
        assert_eq!(out.resolved, ResolvedBackend::Cpu);
        assert!(!out.fallback_occurred);
        assert!(!attempted, "CPU request must not attempt GPU initialization");
    }

    // ---- Strict GPU behaviour (hardening pass, item 3) ----

    #[test]
    fn strict_numerics_rejects_explicit_gpu_at_startup() {
        let mut attempted = false;
        let out = resolve_backend_selection_with_numerics(ComputeOffload::Gpu, true, || {
            attempted = true;
            Ok(())
        });
        assert_eq!(out, Err(BackendResolutionError::StrictGpuUnsupported));
        assert!(
            !attempted,
            "unsupported strict-gpu combination must fail before device init"
        );
    }

    #[test]
    fn strict_numerics_auto_resolves_to_cpu_with_reason() {
        let mut attempted = false;
        let out = resolve_backend_selection_with_numerics(ComputeOffload::Auto, true, || {
            attempted = true;
            Ok(())
        })
        .unwrap();
        assert_eq!(out.resolved, ResolvedBackend::Cpu);
        assert!(!out.fallback_occurred, "strict auto->cpu is a resolution, not a fallback");
        assert!(out.reason.is_some(), "the auto->cpu reason must be recorded");
        assert!(!attempted, "strict auto must not attempt GPU initialization");
    }

    #[test]
    fn hybrid_resolves_to_cpu_attention_gpu_experts() {
        let out = resolve_backend_selection_with_numerics(ComputeOffload::Hybrid, true, || Ok(()))
            .unwrap();
        assert_eq!(out.resolved, ResolvedBackend::HybridCpuAttentionGpuExperts);
        assert!(out.reason.is_some(), "the hybrid split must be recorded");
    }

    #[test]
    fn hybrid_fails_closed_when_gpu_init_fails() {
        let out = resolve_backend_selection_with_numerics(ComputeOffload::Hybrid, true, || {
            Err("no device".to_string())
        });
        assert!(matches!(
            out,
            Err(BackendResolutionError::GpuUnavailable {
                requested: ComputeOffload::Hybrid,
                ..
            })
        ));
    }

    #[test]
    fn hybrid_rejects_legacy_attention_fallback_policy() {
        let out =
            resolve_backend_selection_with_numerics(ComputeOffload::Hybrid, false, || Ok(()));
        assert_eq!(
            out,
            Err(BackendResolutionError::HybridRequiresStrictAttention)
        );
    }

    #[test]
    fn non_strict_gpu_resolution_resolves_to_gpu() {
        let out = resolve_backend_selection_with_numerics(ComputeOffload::Gpu, false, || Ok(()))
            .unwrap();
        assert_eq!(out.resolved, ResolvedBackend::Gpu);
    }

    #[test]
    fn adapter_policy_prefers_high_performance_adapter() {
        let adapters = vec![
            adapter("integrated", wgpu::Backend::Vulkan, wgpu::DeviceType::IntegratedGpu),
            adapter("discrete", wgpu::Backend::Vulkan, wgpu::DeviceType::DiscreteGpu),
        ];

        let order = select_wgpu_adapter_candidates(&adapters, Some(0), false).unwrap();

        assert_eq!(order, vec![0, 1]);
    }

    #[test]
    fn adapter_policy_falls_back_to_discrete_when_high_performance_is_absent() {
        let adapters = vec![
            adapter("integrated", wgpu::Backend::Vulkan, wgpu::DeviceType::IntegratedGpu),
            adapter("discrete", wgpu::Backend::Vulkan, wgpu::DeviceType::DiscreteGpu),
        ];

        let order = select_wgpu_adapter_candidates(&adapters, None, false).unwrap();

        assert_eq!(order, vec![1, 0]);
    }

    #[test]
    fn adapter_policy_skips_software_high_performance_for_real_gpu() {
        let adapters = vec![
            adapter("llvmpipe", wgpu::Backend::Vulkan, wgpu::DeviceType::Cpu),
            adapter("integrated", wgpu::Backend::Vulkan, wgpu::DeviceType::IntegratedGpu),
        ];

        let order = select_wgpu_adapter_candidates(&adapters, Some(0), false).unwrap();

        assert_eq!(order, vec![1]);
    }

    #[test]
    fn adapter_policy_rejects_only_software_without_opt_in() {
        let adapters = vec![adapter(
            "llvmpipe",
            wgpu::Backend::Vulkan,
            wgpu::DeviceType::Cpu,
        )];

        let err = select_wgpu_adapter_candidates(&adapters, Some(0), false).unwrap_err();

        assert_eq!(err, AdapterSelectionError::OnlySoftware { count: 1 });
    }

    #[test]
    fn adapter_policy_rejects_named_software_renderers_even_when_not_cpu_typed() {
        let adapters = vec![
            adapter("softpipe", wgpu::Backend::Gl, wgpu::DeviceType::Other),
            adapter("swrast", wgpu::Backend::Gl, wgpu::DeviceType::Other),
            adapter("OpenSWR", wgpu::Backend::Gl, wgpu::DeviceType::Other),
        ];

        let err = select_wgpu_adapter_candidates(&adapters, None, false).unwrap_err();

        assert_eq!(err, AdapterSelectionError::OnlySoftware { count: 3 });
    }

    #[test]
    fn adapter_policy_allows_software_when_explicitly_enabled() {
        let adapters = vec![adapter(
            "llvmpipe",
            wgpu::Backend::Vulkan,
            wgpu::DeviceType::Cpu,
        )];

        let order = select_wgpu_adapter_candidates(&adapters, Some(0), true).unwrap();

        assert_eq!(order, vec![0]);
    }

    #[test]
    fn test_candle_matmul_correctness() {
        let backend = CandleBackend::new();
        let a_data = [
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(4.0),
        ];
        let b_data = [
            half::f16::from_f32(5.0),
            half::f16::from_f32(6.0),
            half::f16::from_f32(7.0),
            half::f16::from_f32(8.0),
        ];
        let mut out_data = [half::f16::ZERO; 4];

        let a = TensorView {
            data: &a_data,
            rows: 2,
            cols: 2,
        };
        let b = TensorView {
            data: &b_data,
            rows: 2,
            cols: 2,
        };
        let mut out = TensorViewMut {
            data: &mut out_data,
            rows: 2,
            cols: 2,
        };

        backend.matmul_into(a, b, &mut out).unwrap();

        // Expected:
        // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        assert_eq!(out_data[0].to_f32(), 19.0);
        assert_eq!(out_data[1].to_f32(), 22.0);
        assert_eq!(out_data[2].to_f32(), 43.0);
        assert_eq!(out_data[3].to_f32(), 50.0);
    }

    #[test]
    fn test_candle_swiglu_correctness() {
        let backend = CandleBackend::new();
        let gate_data = [half::f16::from_f32(0.0), half::f16::from_f32(1.0)];
        let up_data = [half::f16::from_f32(2.0), half::f16::from_f32(3.0)];
        let mut out_data = [half::f16::ZERO; 2];

        let gate = TensorView {
            data: &gate_data,
            rows: 1,
            cols: 2,
        };
        let up = TensorView {
            data: &up_data,
            rows: 1,
            cols: 2,
        };
        let mut out = TensorViewMut {
            data: &mut out_data,
            rows: 1,
            cols: 2,
        };

        backend.swiglu_into(gate, up, &mut out).unwrap();

        // Expected:
        // out[0] = silu(0) * 2 = 0 * 2 = 0
        // out[1] = silu(1) * 3 = (1 / (1 + exp(-1))) * 3 = 0.7310586 * 3 = 2.1931758
        assert!((out_data[0].to_f32() - 0.0).abs() < 1e-4);
        assert!((out_data[1].to_f32() - 2.1931758).abs() < 1e-3);
    }

    #[test]
    fn test_candle_softmax_correctness() {
        let backend = CandleBackend::new();
        let mut data = [
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(-1.0),
            half::f16::from_f32(0.0),
            half::f16::from_f32(4.0),
        ];
        let mut out = TensorViewMut {
            data: &mut data,
            rows: 2,
            cols: 3,
        };

        backend.softmax(&mut out).unwrap();

        // Row 1 sum: exp(1-3) + exp(2-3) + exp(3-3) = exp(-2) + exp(-1) + 1.0 = 0.1353 + 0.3679 + 1.0 = 1.5032
        // Row 1 values: exp(-2)/1.5032 = 0.0900, exp(-1)/1.5032 = 0.2447, 1.0/1.5032 = 0.6653
        // Sum of Row 1 should be 1.0
        let sum1 = data[0].to_f32() + data[1].to_f32() + data[2].to_f32();
        assert!((sum1 - 1.0).abs() < 1e-3);

        // Row 2 sum: exp(-1-4) + exp(0-4) + exp(4-4) = exp(-5) + exp(-4) + 1.0 = 0.0067 + 0.0183 + 1.0 = 1.0250
        // Sum of Row 2 should be 1.0
        let sum2 = data[3].to_f32() + data[4].to_f32() + data[5].to_f32();
        assert!((sum2 - 1.0).abs() < 1e-3);
    }
}

#[cfg(test)]
mod q4_0_shader_logic_tests {
    //! Host-side mirror of `wgpu_shaders/matmul_q4_0.wgsl`.
    //!
    //! The riskiest part of the inline-dequant GEMV shader is the byte
    //! arithmetic: 18-byte Q4_0 blocks bound as `array<u32>`, per-byte
    //! extraction with shifts, the f16 scale decode and the
    //! low-nibble-first weight order. These tests re-implement that
    //! exact logic in Rust (keep in sync with the WGSL!) and check it
    //! against the canonical CPU dequantiser
    //! [`crate::inference::dequantize_q4_0_block`], so a nibble-order
    //! or offset mistake in the shader's math shows up in CI without
    //! needing a GPU adapter.

    use super::MATMUL_Q4_0_SHADER;
    use crate::inference::{
        dequantize_q4_0_block, quantize_q4_0_block, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS,
    };

    /// Mirror of the WGSL `read_byte` helper.
    fn read_byte(w: &[u32], off: usize) -> u32 {
        (w[off >> 2] >> ((off & 3) * 8)) & 0xff
    }

    /// Pack a little-endian byte stream into the `array<u32>` view the
    /// shader binds, zero-padding to a 4-byte boundary exactly like
    /// `build_expert_entry_q4_0` does.
    fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
        let mut padded = bytes.to_vec();
        padded.resize(bytes.len().div_ceil(4) * 4, 0);
        padded
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Mirror of the WGSL `matmul_q4_0_main` body for one output row.
    fn shader_row_dot(w: &[u32], w_block_off: usize, row: usize, k: usize, x: &[f32]) -> f32 {
        let blocks_per_row = k / Q4_0_BLOCK_ELEMS;
        let mut byte_off = (w_block_off + row * blocks_per_row) * Q4_0_BLOCK_BYTES;
        let mut x_base = 0usize;
        let mut sum = 0.0f32;
        for _ in 0..blocks_per_row {
            let s_lo = read_byte(w, byte_off);
            let s_hi = read_byte(w, byte_off + 1);
            let d = half::f16::from_bits((s_lo | (s_hi << 8)) as u16).to_f32();
            let mut partial = 0.0f32;
            for j in 0..16 {
                let q = read_byte(w, byte_off + 2 + j);
                let w0 = (q & 0xf) as f32 - 8.0;
                let w1 = (q >> 4) as f32 - 8.0;
                partial += w0 * x[x_base + j] + w1 * x[x_base + j + Q4_0_BLOCK_ELEMS / 2];
            }
            sum += d * partial;
            byte_off += Q4_0_BLOCK_BYTES;
            x_base += Q4_0_BLOCK_ELEMS;
        }
        sum
    }

    /// Deterministic pseudo-random weights that exercise the full
    /// nibble range, both signs and varying block scales.
    fn synth_weights(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).max(1);
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state % 2000) as f32 - 1000.0) / 250.0
            })
            .collect()
    }

    #[test]
    fn q4_0_wgsl_parses_and_validates_without_gpu_hardware() {
        let module =
            naga::front::wgsl::parse_str(MATMUL_Q4_0_SHADER).expect("Q4_0 WGSL must parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("Q4_0 WGSL must validate");
        assert!(module
            .entry_points
            .iter()
            .any(|entry| entry.name == "matmul_q4_0_main"));
    }

    /// Quantise an `m × k` row-major matrix into a tight Q4_0 block
    /// stream (rows start on block boundaries because `k % 32 == 0`).
    fn quantize_matrix(weights: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        for chunk in weights.chunks(Q4_0_BLOCK_ELEMS) {
            let mut blk = [0u8; Q4_0_BLOCK_BYTES];
            quantize_q4_0_block(chunk, &mut blk);
            out.extend_from_slice(&blk);
        }
        out
    }

    #[test]
    fn shader_byte_extraction_matches_canonical_block_dequant() {
        // One block whose start lands on a *non*-4-byte-aligned offset
        // (block 1 starts at byte 18), the case the `array<u32>` shift
        // logic exists for.
        let src_a = synth_weights(Q4_0_BLOCK_ELEMS, 7);
        let src_b = synth_weights(Q4_0_BLOCK_ELEMS, 11);
        let mut bytes = Vec::new();
        for src in [&src_a, &src_b] {
            let mut blk = [0u8; Q4_0_BLOCK_BYTES];
            quantize_q4_0_block(src, &mut blk);
            bytes.extend_from_slice(&blk);
        }
        let words = bytes_to_words(&bytes);

        for (bi, range) in [(0usize, 0..Q4_0_BLOCK_BYTES), (1, Q4_0_BLOCK_BYTES..2 * Q4_0_BLOCK_BYTES)] {
            let mut expected = [0.0f32; Q4_0_BLOCK_ELEMS];
            dequantize_q4_0_block(&bytes[range], &mut expected);
            // Dot with a one-hot x isolates each dequantised weight.
            for i in 0..Q4_0_BLOCK_ELEMS {
                let mut x = vec![0.0f32; Q4_0_BLOCK_ELEMS];
                x[i] = 1.0;
                let got = shader_row_dot(&words, bi, 0, Q4_0_BLOCK_ELEMS, &x);
                assert!(
                    (got - expected[i]).abs() < 1e-6,
                    "block {bi} elem {i}: shader logic {got} != canonical {expected:?}"
                );
            }
        }
    }

    #[test]
    fn shader_host_mirror_obeys_independent_ggml_nibble_fixture() {
        let mut block = [0u8; Q4_0_BLOCK_BYTES];
        block[..2].copy_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
        for (j, byte) in block[2..].iter_mut().enumerate() {
            *byte = j as u8 | ((15 - j as u8) << 4);
        }
        let words = bytes_to_words(&block);
        let x: Vec<f32> = (0..Q4_0_BLOCK_ELEMS)
            .map(|i| (i as f32 - 11.0) / 7.0)
            .collect();
        let expected: f32 = (0..16)
            .map(|j| 0.25 * (j as f32 - 8.0) * x[j])
            .chain((0..16).map(|j| 0.25 * (7.0 - j as f32) * x[j + 16]))
            .sum();
        let got = shader_row_dot(&words, 0, 0, Q4_0_BLOCK_ELEMS, &x);
        assert!((got - expected).abs() < 1e-6, "{got} != {expected}");
    }

    #[test]
    fn shader_gemv_matches_cpu_dequant_gemv_with_block_offset() {
        // Small m × k matrix behind a non-zero `w_block_off`, mimicking
        // the up/down projections inside the packed [gate|up|down]
        // expert buffer.
        let (m, k) = (4usize, 64usize);
        let lead_blocks = 3usize; // "gate" blocks preceding this projection
        let lead = synth_weights(lead_blocks * Q4_0_BLOCK_ELEMS, 23);
        let mat = synth_weights(m * k, 42);
        let x = synth_weights(k, 99);

        let mut bytes = quantize_matrix(&lead);
        bytes.extend_from_slice(&quantize_matrix(&mat));
        let words = bytes_to_words(&bytes);

        // Expected: canonical block dequant, then a plain dot per row.
        let mat_bytes = quantize_matrix(&mat);
        let mut dequant = vec![0.0f32; m * k];
        for (b, blk) in mat_bytes.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
            dequantize_q4_0_block(blk, &mut dequant[b * Q4_0_BLOCK_ELEMS..(b + 1) * Q4_0_BLOCK_ELEMS]);
        }
        for row in 0..m {
            let expected: f32 = (0..k).map(|c| dequant[row * k + c] * x[c]).sum();
            let got = shader_row_dot(&words, lead_blocks, row, k, &x);
            assert!(
                (got - expected).abs() < 1e-4 * expected.abs().max(1.0),
                "row {row}: shader logic {got} != cpu {expected}"
            );
        }
    }
}
