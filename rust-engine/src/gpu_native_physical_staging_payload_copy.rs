//! Diagnostic-only host-copy discriminator and one-arm production attribution.
//! GPU resources here are isolated from inference; every bounded staging batch
//! is submitted and drained outside its wall/copy timers before the next batch.
use super::*;
use crate::backend::gpu_native::{
    CheckedPhysicalQ4ExpertSlot, GpuNativeQ4ExpertGeometry, PhysicalSlotObservation,
};
use crate::engine::GpuNativePhysicalInstallConcurrencyQualificationSnapshot as Snapshot;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

pub(crate) const SCHEMA: &str = "mer.gpu-native-physical-staging-payload-copy.v1";
pub(crate) const MODE: &str = "diagnose-gpu-native-physical-staging-payload-copy";
const WIDTHS: [usize; 4] = [1, 2, 4, 8];
const PAYLOAD: usize = 2_654_208;
const STRIDE: usize = 2_654_212;
const OFFSET: usize = 4;
const MATERIALITY: f64 = 0.10;
const MAX_ITERATIONS: usize = 1000;

type DiagnosticResult<T> = Result<T, String>;

#[derive(Clone, Debug)]
pub(crate) struct Args {
    pub(crate) common: CommandArgs,
    pub(crate) standalone_only: bool,
    pub(crate) iterations: usize,
    pub(crate) warmup_iterations: usize,
}

#[derive(Debug, Serialize)]
struct Geometry {
    payload_offset_bytes: usize,
    payload_bytes: usize,
    slot_stride_bytes: usize,
    tail_padding_bytes: usize,
}

fn geometry() -> DiagnosticResult<(GpuNativeQ4ExpertGeometry, Geometry)> {
    let g = GpuNativeQ4ExpertGeometry::try_new(2048, 768, 128, 8).map_err(|e| e.to_string())?;
    let end = g
        .payload_offset_bytes()
        .checked_add(g.logical_expert_bytes())
        .ok_or("geometry overflow")?;
    let tail = g
        .slot_stride_bytes()
        .checked_sub(end)
        .ok_or("geometry coverage underflow")?;
    if (
        g.payload_offset_bytes(),
        g.logical_expert_bytes(),
        g.slot_stride_bytes(),
        tail,
    ) != (OFFSET, PAYLOAD, STRIDE, 0)
    {
        return Err("frozen geometry mismatch".into());
    }
    Ok((
        g,
        Geometry {
            payload_offset_bytes: OFFSET,
            payload_bytes: PAYLOAD,
            slot_stride_bytes: STRIDE,
            tail_padding_bytes: tail,
        },
    ))
}

/// Versioned, platform-independent byte sequence; no randomness or model data.
fn payload() -> Vec<u8> {
    (0..PAYLOAD)
        .map(|i| ((i as u64 * 131 + (i as u64 >> 8) * 17 + 29) & 255) as u8)
        .collect()
}

fn add(a: u64, b: u64) -> DiagnosticResult<u64> {
    a.checked_add(b)
        .ok_or_else(|| "timing/byte accounting overflow".into())
}
fn ns(d: Duration) -> DiagnosticResult<u64> {
    u64::try_from(d.as_nanos()).map_err(|_| "duration overflow".into())
}
fn gbps(bytes: u64, nanos: u64) -> Option<f64> {
    (nanos != 0).then(|| bytes as f64 / nanos as f64)
}
fn ratio(a: f64, b: f64) -> Option<f64> {
    (a.is_finite() && b.is_finite() && b > 0.0).then(|| a / b)
}

#[derive(Default)]
struct Active {
    jobs: AtomicUsize,
    max: AtomicUsize,
}
struct Job<'a>(&'a Active);
impl Active {
    fn enter(&self) -> Job<'_> {
        let n = self.jobs.fetch_add(1, Ordering::AcqRel) + 1;
        self.max.fetch_max(n, Ordering::Relaxed);
        Job(self)
    }
}
impl Drop for Job<'_> {
    fn drop(&mut self) {
        self.0.jobs.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Clone, Debug, Default)]
struct JobTimes {
    copy_ns: u64,
    validation_ns: u64,
    epoch_ns: u64,
    acquire_ns: u64,
    drop_ns: u64,
}
impl JobTimes {
    fn observed(o: PhysicalSlotObservation) -> DiagnosticResult<Self> {
        Ok(Self {
            copy_ns: ns(o.payload_copy)?,
            validation_ns: ns(o.validation)?,
            epoch_ns: ns(o.epoch_write)?,
            ..Self::default()
        })
    }
}

/// Destination storage is supplied by the caller. There are no allocations in
/// the observed fill or its timed copy; the exact production coverage check is
/// shared. black_box is outside all subphase timers.
fn ram_copy(
    g: GpuNativeQ4ExpertGeometry,
    source: &[u8],
    destination: &mut [u8],
) -> DiagnosticResult<JobTimes> {
    let checked = CheckedPhysicalQ4ExpertSlot::new(g, 0, 1, source).map_err(|e| e.to_string())?;
    let observed = checked
        .fill_complete_overwrite_no_zero_observed(destination)
        .map_err(|e| e.to_string())?;
    std::hint::black_box(destination);
    JobTimes::observed(observed)
}

#[derive(Clone, Debug, Serialize)]
struct Batch {
    iteration: usize,
    operations: u64,
    payload_bytes: u64,
    sum_individual_copy_ns: u64,
    sum_individual_copy_us: f64,
    validation_us: f64,
    epoch_write_us: f64,
    wall_ns: u64,
    wall_us: f64,
    view_acquisition_us: f64,
    view_drop_scheduling_us: f64,
    submit_and_drain_us: f64,
    max_simultaneous_jobs_observed: usize,
    failures: Vec<String>,
    accounting_errors: Vec<String>,
}

/// Exactly one job per supplied destination; inline singleton and shared
/// process-wide Rayon for wider batches, matching physical staging's style.
fn batch<T: Send, F>(iteration: usize, destinations: &mut [T], job: F) -> Batch
where
    F: Fn(usize, &mut T) -> DiagnosticResult<JobTimes> + Sync + Send,
{
    let active = Active::default();
    let run = |(i, destination): (usize, &mut T)| {
        let _guard = active.enter();
        job(i, destination)
    };
    let started = Instant::now();
    let results: Vec<_> = if destinations.len() == 1 {
        destinations.iter_mut().enumerate().map(run).collect()
    } else {
        destinations.par_iter_mut().enumerate().map(run).collect()
    };
    let wall = started.elapsed();
    let mut b = Batch {
        iteration,
        operations: 0,
        payload_bytes: 0,
        sum_individual_copy_ns: 0,
        sum_individual_copy_us: 0.0,
        validation_us: 0.0,
        epoch_write_us: 0.0,
        wall_ns: 0,
        wall_us: 0.0,
        view_acquisition_us: 0.0,
        view_drop_scheduling_us: 0.0,
        submit_and_drain_us: 0.0,
        max_simultaneous_jobs_observed: active.max.load(Ordering::Relaxed),
        failures: vec![],
        accounting_errors: vec![],
    };
    let mut validation = 0;
    let mut epoch = 0;
    let mut acquire = 0;
    let mut drop_time = 0;
    let accounting = (|| -> DiagnosticResult<()> {
        b.wall_ns = ns(wall)?;
        for result in results {
            match result {
                Err(e) => b.failures.push(e),
                Ok(t) => {
                    b.operations = add(b.operations, 1)?;
                    b.payload_bytes = add(b.payload_bytes, PAYLOAD as u64)?;
                    b.sum_individual_copy_ns = add(b.sum_individual_copy_ns, t.copy_ns)?;
                    validation = add(validation, t.validation_ns)?;
                    epoch = add(epoch, t.epoch_ns)?;
                    acquire = add(acquire, t.acquire_ns)?;
                    drop_time = add(drop_time, t.drop_ns)?;
                }
            }
        }
        if active.jobs.load(Ordering::Relaxed) != 0
            || b.max_simultaneous_jobs_observed > destinations.len()
        {
            return Err("active staging jobs exceeded requested width or leaked".into());
        }
        Ok(())
    })();
    if let Err(e) = accounting {
        b.accounting_errors.push(e);
    }
    b.wall_us = b.wall_ns as f64 / 1000.0;
    b.sum_individual_copy_us = b.sum_individual_copy_ns as f64 / 1000.0;
    b.validation_us = validation as f64 / 1000.0;
    b.epoch_write_us = epoch as f64 / 1000.0;
    b.view_acquisition_us = acquire as f64 / 1000.0;
    b.view_drop_scheduling_us = drop_time as f64 / 1000.0;
    b
}

#[derive(Debug, Serialize)]
struct Case {
    destination: &'static str,
    width: usize,
    operations: u64,
    bytes_per_operation: usize,
    total_payload_bytes: u64,
    sum_individual_copy_us: f64,
    batch_iteration_wall_us: f64,
    effective_aggregate_gbps_from_wall: Option<f64>,
    effective_summed_copy_gbps: Option<f64>,
    view_acquisition_us: Option<f64>,
    view_drop_scheduling_us: Option<f64>,
    epoch_write_us: f64,
    validation_us: f64,
    submit_and_drain_us: Option<f64>,
    max_simultaneous_jobs_observed: usize,
    failures: Vec<String>,
    accounting_errors: Vec<String>,
    destination_payload_sha256: Option<String>,
    batches: Vec<Batch>,
}

fn summarize(
    destination: &'static str,
    width: usize,
    batches: Vec<Batch>,
    checksum: Option<String>,
) -> DiagnosticResult<Case> {
    let mut ops = 0;
    let mut bytes = 0;
    let mut copy = 0;
    let mut wall = 0;
    for b in &batches {
        ops = add(ops, b.operations)?;
        bytes = add(bytes, b.payload_bytes)?;
        copy = add(copy, b.sum_individual_copy_ns)?;
        wall = add(wall, b.wall_ns)?;
    }
    let staging = destination == "wgpu-staging-view";
    let mut c = Case {
        destination,
        width,
        operations: ops,
        bytes_per_operation: PAYLOAD,
        total_payload_bytes: bytes,
        sum_individual_copy_us: copy as f64 / 1000.0,
        batch_iteration_wall_us: wall as f64 / 1000.0,
        effective_aggregate_gbps_from_wall: gbps(bytes, wall),
        effective_summed_copy_gbps: gbps(bytes, copy),
        view_acquisition_us: staging.then(|| batches.iter().map(|b| b.view_acquisition_us).sum()),
        view_drop_scheduling_us: staging
            .then(|| batches.iter().map(|b| b.view_drop_scheduling_us).sum()),
        epoch_write_us: batches.iter().map(|b| b.epoch_write_us).sum(),
        validation_us: batches.iter().map(|b| b.validation_us).sum(),
        submit_and_drain_us: staging.then(|| batches.iter().map(|b| b.submit_and_drain_us).sum()),
        max_simultaneous_jobs_observed: batches
            .iter()
            .map(|b| b.max_simultaneous_jobs_observed)
            .max()
            .unwrap_or(0),
        failures: batches.iter().flat_map(|b| b.failures.clone()).collect(),
        accounting_errors: batches
            .iter()
            .flat_map(|b| b.accounting_errors.clone())
            .collect(),
        destination_payload_sha256: checksum,
        batches,
    };
    if (width as u64).checked_mul(c.batches.len() as u64) != Some(ops)
        || ops.checked_mul(PAYLOAD as u64) != Some(bytes)
        || c.max_simultaneous_jobs_observed > width
    {
        c.accounting_errors
            .push("operations/bytes/width mismatch".into());
    }
    if copy == 0 || wall == 0 {
        c.accounting_errors
            .push("zero elapsed time cannot define throughput".into());
    }
    Ok(c)
}

#[derive(Debug, Serialize)]
struct WidthComparison {
    width: usize,
    wgpu_to_ram_summed_copy_gbps_ratio: Option<f64>,
    wgpu_to_ram_wall_gbps_ratio: Option<f64>,
    paired_copy_gbps_ratios: Vec<f64>,
    paired_ratio_p10: Option<f64>,
    paired_ratio_p90: Option<f64>,
}
fn comparison_for(r: &Case, w: &Case) -> WidthComparison {
    let ratios: Vec<f64> = r
        .batches
        .iter()
        .zip(&w.batches)
        .filter_map(|(r, w)| {
            if r.operations != w.operations || !r.failures.is_empty() || !w.failures.is_empty() {
                return None;
            }
            ratio(
                gbps(w.payload_bytes, w.sum_individual_copy_ns)?,
                gbps(r.payload_bytes, r.sum_individual_copy_ns)?,
            )
        })
        .collect();
    let mut sorted = ratios.clone();
    sorted.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        (!sorted.is_empty()).then(|| sorted[((sorted.len() - 1) as f64 * p).ceil() as usize])
    };
    WidthComparison {
        width: r.width,
        wgpu_to_ram_summed_copy_gbps_ratio: w
            .effective_summed_copy_gbps
            .zip(r.effective_summed_copy_gbps)
            .and_then(|(a, b)| ratio(a, b)),
        wgpu_to_ram_wall_gbps_ratio: w
            .effective_aggregate_gbps_from_wall
            .zip(r.effective_aggregate_gbps_from_wall)
            .and_then(|(a, b)| ratio(a, b)),
        paired_ratio_p10: percentile(0.1),
        paired_ratio_p90: percentile(0.9),
        paired_copy_gbps_ratios: ratios,
    }
}

#[derive(Debug, Serialize)]
struct AdapterIdentity {
    name: String,
    vendor: u32,
    device: u32,
    device_type: String,
    backend: String,
    driver: String,
    driver_info: String,
}
#[derive(Debug, Serialize)]
struct Standalone {
    adapter: AdapterIdentity,
    authoritative_l4: bool,
    rayon_num_threads: usize,
    max_pending_staging_writes: usize,
    max_pending_staging_bytes: usize,
    source_pattern: &'static str,
    source_payload_sha256: String,
    warmup_iterations_per_width_and_destination: usize,
    measured_iterations_per_width_and_destination: usize,
    order: &'static str,
    wall_definition: &'static str,
    ram: Vec<Case>,
    wgpu_staging: Vec<Case>,
    comparisons: Vec<WidthComparison>,
}

async fn standalone(args: &Args) -> DiagnosticResult<Standalone> {
    let (g, _) = geometry()?;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: if args.common.expected_adapter_name == "NVIDIA L4" {
            wgpu::Backends::VULKAN
        } else {
            wgpu::Backends::all()
        },
        ..Default::default()
    });
    let adapter = instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .find(|a| a.get_info().name == args.common.expected_adapter_name)
        .ok_or_else(|| {
            format!(
                "expected adapter {:?} unavailable",
                args.common.expected_adapter_name
            )
        })?;
    let info = adapter.get_info();
    let authoritative_l4 = cfg!(target_os = "linux")
        && info.name == "NVIDIA L4"
        && info.vendor == 0x10de
        && info.device_type == wgpu::DeviceType::DiscreteGpu
        && info.backend == wgpu::Backend::Vulkan;
    if !args.standalone_only && !authoritative_l4 {
        return Err("real inference attribution requires Linux NVIDIA L4/Vulkan".into());
    }
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some(MODE),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .map_err(|e| e.to_string())?;
    let gpu_errors = Arc::new(parking_lot::Mutex::new(Vec::<String>::new()));
    let errors = gpu_errors.clone();
    device.on_uncaptured_error(Box::new(move |error| errors.lock().push(error.to_string())));
    let source = payload();
    let source_hash = crate::greedy_parity::sha256_hex(&source);
    let max_bytes = STRIDE
        .checked_mul(*WIDTHS.last().unwrap())
        .ok_or("bounded memory overflow")?;
    let target = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("payload-copy-diagnostic-only"),
        size: u64::try_from(max_bytes).map_err(|_| "buffer size overflow")?,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut result = Standalone {
        adapter: AdapterIdentity { name: info.name, vendor: info.vendor, device: info.device, device_type: format!("{:?}", info.device_type), backend: format!("{:?}", info.backend), driver: info.driver, driver_info: info.driver_info },
        authoritative_l4, rayon_num_threads: rayon::current_num_threads(), max_pending_staging_writes: 8,
        max_pending_staging_bytes: max_bytes, source_pattern: "v1: byte[i] = (i*131 + (i>>8)*17 + 29) mod 256",
        source_payload_sha256: source_hash.clone(), warmup_iterations_per_width_and_destination: args.warmup_iterations,
        measured_iterations_per_width_and_destination: args.iterations,
        order: "alternate RAM-first and WGPU-first by iteration at each ascending width; warmups excluded",
        wall_definition: "host job dispatch/join plus validation, epoch, copy and (WGPU only) acquisition/drop; excludes buffer allocation, checksum, submit and drain; GB/s is decimal host throughput, not DMA bandwidth",
        ram: vec![], wgpu_staging: vec![], comparisons: vec![],
    };
    for width in WIDTHS {
        let mut ram = vec![vec![0xa5; STRIDE]; width];
        let mut slots = vec![(); width];
        let mut ram_batches = Vec::with_capacity(args.iterations);
        let mut wgpu_batches = Vec::with_capacity(args.iterations);
        let iterations = args
            .warmup_iterations
            .checked_add(args.iterations)
            .ok_or("iteration overflow")?;
        for iteration in 0..iterations {
            for ram_first in if iteration % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            } {
                let mut b = if ram_first {
                    batch(iteration, &mut ram, |_, destination| {
                        ram_copy(g, &source, destination)
                    })
                } else {
                    batch(iteration, &mut slots, |i, _| {
                        let checked = CheckedPhysicalQ4ExpertSlot::new(g, 0, 1, &source)
                            .map_err(|e| e.to_string())?;
                        let offset = i
                            .checked_mul(STRIDE)
                            .and_then(|v| u64::try_from(v).ok())
                            .ok_or("offset overflow")?;
                        let started = Instant::now();
                        let mut view = queue
                            .write_buffer_with(
                                &target,
                                offset,
                                wgpu::BufferSize::new(STRIDE as u64).unwrap(),
                            )
                            .ok_or("write_buffer_with unavailable")?;
                        let acquire_ns = ns(started.elapsed())?;
                        let observed =
                            checked.fill_complete_overwrite_no_zero_observed(view.as_mut());
                        let started = Instant::now();
                        drop(view);
                        let drop_ns = ns(started.elapsed())?;
                        let mut times = JobTimes::observed(observed.map_err(|e| e.to_string())?)?;
                        times.acquire_ns = acquire_ns;
                        times.drop_ns = drop_ns;
                        Ok(times)
                    })
                };
                if ram_first {
                    // Observable, full byte-level check after every RAM phase,
                    // outside the wall and copy timers, without a GPU readback.
                    for destination in &ram {
                        if &destination[..OFFSET] != 1u32.to_le_bytes().as_slice()
                            || crate::greedy_parity::sha256_hex(&destination[OFFSET..])
                                != source_hash
                        {
                            b.failures.push("RAM destination content mismatch".into());
                        }
                    }
                } else {
                    // This isolated queue holds <= width scheduled writes. Drain
                    // every batch, including failed/warmup batches, before reuse.
                    let started = Instant::now();
                    let submission = queue.submit(std::iter::empty());
                    device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
                    b.submit_and_drain_us = ns(started.elapsed())? as f64 / 1000.0;
                    b.failures.extend(gpu_errors.lock().drain(..));
                }
                if iteration < args.warmup_iterations {
                    if !b.failures.is_empty() || !b.accounting_errors.is_empty() {
                        return Err(format!(
                            "warmup failed: {:?} {:?}",
                            b.failures, b.accounting_errors
                        ));
                    }
                } else if ram_first {
                    ram_batches.push(b);
                } else {
                    wgpu_batches.push(b);
                }
            }
        }
        let r = summarize("plain-ram", width, ram_batches, Some(source_hash.clone()))?;
        let w = summarize("wgpu-staging-view", width, wgpu_batches, None)?;
        result.comparisons.push(comparison_for(&r, &w));
        result.ram.push(r);
        result.wgpu_staging.push(w);
    }
    Ok(result)
}

#[derive(Debug, Serialize)]
struct ProductionTotals {
    install_count: u64,
    payload_bytes_copied: u64,
    epoch_bytes_written: u64,
    staged_bytes: u64,
    physical_slot_validation_us: u64,
    physical_slot_epoch_write_us: u64,
    physical_slot_payload_copy_us: u64,
    physical_slot_prepare_us: u64,
    physical_queue_staging_us: u64,
    physical_slot_prepare_residual_us: u64,
    payload_copy_effective_gbps: Option<f64>,
    payload_copy_share_of_prepare: Option<f64>,
    residual_share_of_prepare: Option<f64>,
    subphase_observations: u64,
    accounting_valid: bool,
}
fn production_totals(s: &Snapshot) -> ProductionTotals {
    let parts = s
        .physical_slot_validation_us
        .checked_add(s.physical_slot_epoch_write_us)
        .and_then(|v| v.checked_add(s.physical_slot_payload_copy_us));
    ProductionTotals {
        install_count: s.physical_install_completions,
        payload_bytes_copied: s.physical_slot_payload_copy_bytes,
        epoch_bytes_written: s.physical_slot_epoch_write_bytes,
        staged_bytes: s.physical_slot_bytes_staged,
        physical_slot_validation_us: s.physical_slot_validation_us,
        physical_slot_epoch_write_us: s.physical_slot_epoch_write_us,
        physical_slot_payload_copy_us: s.physical_slot_payload_copy_us,
        physical_slot_prepare_us: s.physical_slot_prepare_us,
        physical_queue_staging_us: s.physical_queue_staging_us,
        physical_slot_prepare_residual_us: s.physical_slot_prepare_residual_us,
        payload_copy_effective_gbps: ratio(
            s.physical_slot_payload_copy_bytes as f64,
            s.physical_slot_payload_copy_us as f64 * 1000.0,
        ),
        payload_copy_share_of_prepare: ratio(
            s.physical_slot_payload_copy_us as f64,
            s.physical_slot_prepare_us as f64,
        ),
        residual_share_of_prepare: ratio(
            s.physical_slot_prepare_residual_us as f64,
            s.physical_slot_prepare_us as f64,
        ),
        subphase_observations: s.physical_slot_subphase_observations,
        accounting_valid: s.physical_install_completions > 0
            && s.physical_slot_subphase_observations == s.physical_install_completions
            && zero_fill_production::bytes_exact(s)
            && zero_fill_production::failures_zero(s)
            && zero_fill_production::timing_accounting_exact(s)
            && parts.and_then(|v| v.checked_add(s.physical_slot_prepare_residual_us))
                == Some(s.physical_slot_prepare_us),
    }
}

#[derive(Debug, Serialize)]
struct RealInference {
    frozen_workload: FrozenWorkload,
    provenance: BenchmarkProvenance,
    production_arm: ConcurrencyArmReport,
    warmup_attribution: Option<ProductionTotals>,
    measured_production_attribution: Option<ProductionTotals>,
    accounting_valid: bool,
}
fn phase_valid(
    s: Option<&Snapshot>,
    p: Option<&GpuNativeProductionPhysicalInstallSnapshot>,
    w: Option<&ArmWorkEvidence>,
    source: Option<&ProductionDemandSourceSnapshot>,
) -> bool {
    match (s, p, w, source) {
        (Some(s), Some(p), Some(w), Some(source)) => production_totals(s).accounting_valid
            && s.arm == crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm::ProductionNoZeroFillTreatment
            && s.treatment_uses_ordinary_production_path && s.physical_slot_zero_fill_bytes == 0
            && s.full_slot_vec_materializations == 0
            && zero_fill_production::install_accounting_exact(s, p)
            && zero_fill_production::work_errors_zero(w) && production_safety_zero(source)
            && s.physical_install_completions == w.gpu_native_residency.ram_to_vram_installs,
        _ => false,
    }
}
async fn real_inference(args: &Args) -> Result<RealInference, Box<dyn std::error::Error>> {
    if args.common.expected_adapter_name != "NVIDIA L4" || !cfg!(target_os = "linux") {
        return Err("frozen real attribution requires Linux and exact NVIDIA L4".into());
    }
    let prepared = prepare(&args.common)?;
    if prepared.spec.cfg.gpu_cache.vram_capacity_mb != 2048 {
        return Err("frozen real attribution requires 2 GiB managed expert VRAM".into());
    }
    let run = run_physical_install_arm_inner(
        &prepared,
        &args.common,
        PhysicalInstallQualificationRun::ZeroFillProduction(
            crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm::ProductionNoZeroFillTreatment,
        ),
        Some(MODE),
    )
    .await?;
    let production_arm = ConcurrencyArmReport {
        common: run.common,
        warmup_mechanism: run.warmup_concurrency,
        mechanism: run.concurrency,
    };
    let warmup_attribution = production_arm
        .warmup_mechanism
        .as_ref()
        .map(production_totals);
    let measured_production_attribution = production_arm.mechanism.as_ref().map(production_totals);
    let c = &production_arm.common;
    let accounting_valid = c.complete
        && c.failure.is_none()
        && c.warmup_results.len() == FROZEN_WARMUP_RUNS
        && c.warmup_results
            .iter()
            .all(|r| r.generated_tokens == FROZEN_OUTPUT_TOKENS)
        && generated_results(c).len() == FROZEN_MEASURED_RUNS
        && generated_results(c)
            .iter()
            .all(|r| r.generated_tokens == FROZEN_OUTPUT_TOKENS)
        && phase_valid(
            production_arm.warmup_mechanism.as_ref(),
            c.warmup_production_physical_install.as_ref(),
            c.warmup_work.as_ref(),
            c.warmup_production.as_ref(),
        )
        && phase_valid(
            production_arm.mechanism.as_ref(),
            c.production_physical_install.as_ref(),
            c.work.as_ref(),
            c.production.as_ref(),
        );
    Ok(RealInference {
        frozen_workload: frozen_workload(args.common.expected_adapter_name.clone()),
        provenance: prepared.provenance,
        production_arm,
        warmup_attribution,
        measured_production_attribution,
        accounting_valid,
    })
}

fn classify(s: &Standalone, real: Option<&ProductionTotals>) -> &'static str {
    if s.ram
        .iter()
        .chain(&s.wgpu_staging)
        .any(|c| !c.failures.is_empty() || !c.accounting_errors.is_empty())
    {
        return "no-clear-discriminator";
    }
    if s.comparisons.len() != WIDTHS.len()
        || s.comparisons
            .iter()
            .any(|c| c.paired_copy_gbps_ratios.len() < 10)
    {
        return "no-clear-discriminator";
    }
    // Demand a >=10% difference even at paired p90, at >=3 of 4 widths,
    // with at least 10 paired iterations. Raw batches remain authoritative.
    if s.comparisons
        .iter()
        .filter(|c| {
            c.paired_copy_gbps_ratios.len() >= 10
                && c.paired_ratio_p90.is_some_and(|r| r <= 1.0 - MATERIALITY)
                && c.wgpu_to_ram_summed_copy_gbps_ratio
                    .is_some_and(|r| r <= 1.0 - MATERIALITY)
        })
        .count()
        >= 3
    {
        return "staging-view-memory-slower-than-ram";
    }
    let similar = s.comparisons.iter().all(|c| {
        c.paired_ratio_p10.is_some_and(|r| r >= 1.0 - MATERIALITY)
            && c.paired_ratio_p90.is_some_and(|r| r <= 1.0 + MATERIALITY)
            && c.wgpu_to_ram_summed_copy_gbps_ratio
                .is_some_and(|r| (1.0 - MATERIALITY..=1.0 + MATERIALITY).contains(&r))
    });
    let scaling = |cases: &[Case]| {
        cases.first().zip(cases.last()).and_then(|(a, b)| {
            ratio(
                b.effective_aggregate_gbps_from_wall?,
                a.effective_aggregate_gbps_from_wall?,
            )
        })
    };
    if similar
        && s.wgpu_staging
            .last()
            .is_some_and(|c| c.max_simultaneous_jobs_observed >= 2)
        && scaling(&s.wgpu_staging).is_some_and(|r| r >= 1.25)
        && scaling(&s.ram).is_some_and(|r| r >= 1.25)
    {
        return "concurrency-amortization-dominant";
    }
    if similar
        && real.is_some_and(|r| {
            r.accounting_valid && r.payload_copy_share_of_prepare.is_some_and(|r| r >= 0.8)
        })
    {
        return "host-memcpy-bandwidth-dominant";
    }
    "no-clear-discriminator"
}

#[derive(Debug, Serialize)]
struct Report {
    schema: &'static str,
    mode: &'static str,
    diagnostic_only: bool,
    build_provenance: BuildProvenance,
    tree_sha: Option<String>,
    tree_sha_note: &'static str,
    executable_identity: Option<(PathBuf, String)>,
    geometry: Option<Geometry>,
    requested_widths: [usize; 4],
    standalone_only: bool,
    materiality_threshold_fraction: f64,
    classification_rule: &'static str,
    timing_definition: &'static str,
    standalone: Option<Standalone>,
    real_inference: Option<RealInference>,
    complete: bool,
    failures: Vec<String>,
    diagnostic_classification: &'static str,
}

pub(crate) async fn run_command(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = Report {
        schema: SCHEMA, mode: MODE, diagnostic_only: true, build_provenance: BuildProvenance::embedded(),
        tree_sha: None, tree_sha_note: "existing embedded build provenance does not supply a tree SHA",
        executable_identity: None, geometry: None, requested_widths: WIDTHS, standalone_only: args.standalone_only,
        materiality_threshold_fraction: MATERIALITY,
        classification_rule: "priority: staging slower if aggregate ratio and paired p90 <=0.90 at >=3 widths with >=10 pairs; all classifications need >=10 pairs per width; else concurrency if all aggregate and paired p10/p90 copy ratios in [0.90,1.10] and both width8/width1 wall throughputs >=1.25; else host memcpy if ratios similar and valid real copy share >=0.80; else unclear. This is descriptive evidence, not proof of memory type or an optimization recommendation; raw measurements are authoritative.",
        timing_definition: "production validation includes original checked-writer pre-view preparation plus shared coverage validation; prepare retains historical outer timing. residual=prepare-(validation+epoch+payload), checked; accounting error on underflow/overflow. Production integer microseconds truncate each install subphase (especially epoch); residual includes timer overhead and truncation. Standalone retains nanoseconds and reports fractional microseconds. Decode TPS cannot affect diagnostic completeness.",
        standalone: None, real_inference: None, complete: false, failures: vec![], diagnostic_classification: "no-clear-discriminator",
    };
    let result: Result<(), Box<dyn std::error::Error>> = async {
        if !(1..=MAX_ITERATIONS).contains(&args.iterations) || !(1..=100).contains(&args.warmup_iterations) {
            return Err("iterations must be 1..=1000 and warmup-iterations 1..=100".into());
        }
        report.geometry = Some(geometry()?.1);
        report.executable_identity = Some(crate::current_executable_identity()?);
        report.standalone = Some(standalone(&args).await?);
        if report.standalone.as_ref().unwrap().ram.iter().chain(&report.standalone.as_ref().unwrap().wgpu_staging)
            .any(|c| !c.failures.is_empty() || !c.accounting_errors.is_empty()) { return Err("standalone copy failures/accounting errors; see cases".into()); }
        if !args.standalone_only {
            report.real_inference = Some(real_inference(&args).await?);
            if !report.real_inference.as_ref().unwrap().accounting_valid { return Err("real production attribution incomplete or accounting invalid; see arm evidence".into()); }
        }
        report.diagnostic_classification = classify(report.standalone.as_ref().unwrap(), report.real_inference.as_ref().and_then(|r| r.measured_production_attribution.as_ref()));
        report.complete = true;
        Ok(())
    }.await;
    if let Err(e) = &result {
        report.failures.push(e.to_string());
    }
    emit_report(&report, &args.common.report_out)?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_copy_contract_geometry_payload_identity_and_widths_are_frozen() {
        assert_eq!(SCHEMA, "mer.gpu-native-physical-staging-payload-copy.v1");
        assert_eq!(WIDTHS, [1, 2, 4, 8]);
        let (_, g) = geometry().unwrap();
        assert_eq!(
            (
                g.payload_offset_bytes,
                g.payload_bytes,
                g.slot_stride_bytes,
                g.tail_padding_bytes
            ),
            (4, 2_654_208, 2_654_212, 0)
        );
        let source = payload();
        assert_eq!(source.len(), PAYLOAD);
        assert_eq!(
            crate::greedy_parity::sha256_hex(&source),
            "08a90a4f167d260eea22683314fc19b72d06b5420c01188e5c7e765e09219336"
        );
    }

    #[test]
    fn payload_copy_ram_reuses_preallocated_destination_and_copy_has_no_allocator() {
        let (g, _) = geometry().unwrap();
        let source = payload();
        let mut destination = vec![0xa5; STRIDE];
        let ptr = destination.as_ptr();
        let capacity = destination.capacity();
        for _ in 0..3 {
            ram_copy(g, &source, &mut destination).unwrap();
            assert_eq!(destination.as_ptr(), ptr);
            assert_eq!(destination.capacity(), capacity);
            assert_eq!(&destination[..4], &1u32.to_le_bytes());
            assert_eq!(&destination[4..], source.as_slice());
        }
        let backend = include_str!("backend/gpu_native.rs");
        let observed = backend
            .split("    pub(crate) fn fill_complete_overwrite_no_zero_observed(\n")
            .nth(1)
            .unwrap()
            .split("\nfn validate_q4_expert_uploads")
            .next()
            .unwrap();
        for forbidden in ["vec![", "Vec::", ".to_vec(", ".resize(", "alloc("] {
            assert!(!observed.contains(forbidden));
        }
        assert!(observed.contains(".copy_from_slice("));
    }

    #[test]
    fn payload_copy_jobs_are_bounded_at_every_width_and_failures_join() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        for width in WIDTHS {
            let mut destinations = vec![0; width];
            let b = pool.install(|| {
                batch(0, &mut destinations, |i, d| {
                    *d = i + 1;
                    std::thread::yield_now();
                    Ok(JobTimes {
                        copy_ns: 100,
                        ..Default::default()
                    })
                })
            });
            assert_eq!(b.operations, width as u64);
            assert!((1..=width).contains(&b.max_simultaneous_jobs_observed));
            assert_eq!(destinations, (1..=width).collect::<Vec<_>>());
            assert!(b.accounting_errors.is_empty());
            let failed = pool.install(|| {
                batch(1, &mut destinations, |i, _| {
                    if i == 0 {
                        Err("injected acquisition failure".into())
                    } else {
                        Ok(JobTimes::default())
                    }
                })
            });
            assert_eq!(failed.failures, ["injected acquisition failure"]);
            assert_eq!(failed.operations, width as u64 - 1);
            assert!(failed.max_simultaneous_jobs_observed <= width);
        }
    }

    #[test]
    fn payload_copy_aggregation_reports_overflow_and_work_mismatch() {
        let mut slots = vec![(); 2];
        let b = batch(0, &mut slots, |_, _| {
            Ok(JobTimes {
                copy_ns: u64::MAX,
                ..Default::default()
            })
        });
        assert!(!b.accounting_errors.is_empty());
        let c = summarize("plain-ram", 4, vec![b], None).unwrap();
        assert!(c
            .accounting_errors
            .iter()
            .any(|e| e.contains("operations/bytes/width")));
        assert!(add(u64::MAX, 1).is_err());
        assert_eq!(gbps(2_654_208, 1_000_000), Some(2.654208));
        assert!(gbps(1, 0).is_none());
    }

    fn production_fixture() -> Snapshot {
        let mut s = crate::engine::empty_physical_zero_fill_test_snapshot(
            crate::engine::GpuNativePhysicalInstallConcurrencyQualificationArm::ProductionNoZeroFillTreatment);
        s.physical_install_completions = 3;
        s.physical_slot_subphase_observations = 3;
        s.physical_slot_epoch_write_bytes = 12;
        s.physical_slot_payload_copy_bytes = 3 * PAYLOAD as u64;
        s.physical_slot_bytes_staged = 3 * STRIDE as u64;
        s.physical_bytes_staged = s.physical_slot_bytes_staged;
        s.physical_slot_validation_us = 3;
        s.physical_slot_epoch_write_us = 1;
        s.physical_slot_payload_copy_us = 90;
        s.physical_slot_prepare_residual_us = 6;
        s.physical_slot_prepare_us = 100;
        s.physical_queue_staging_us = 10;
        s.sum_individual_physical_stage_us = 120;
        s.physical_ordered_commit_us = 5;
        s.physical_install_total_us = 125;
        s
    }

    #[test]
    fn payload_copy_production_totals_fail_closed_on_every_accounting_corruption() {
        let s = production_fixture();
        let totals = production_totals(&s);
        assert!(totals.accounting_valid);
        assert_eq!(totals.payload_copy_share_of_prepare, Some(0.9));
        assert_eq!(totals.residual_share_of_prepare, Some(0.06));
        for corrupt in [
            (|s: &mut Snapshot| s.physical_slot_epoch_write_bytes += 1) as fn(&mut Snapshot),
            |s| s.physical_slot_payload_copy_bytes += 1,
            |s| s.physical_slot_bytes_staged += 1,
            |s| s.physical_slot_subphase_observations -= 1,
            |s| s.physical_slot_prepare_residual_us -= 1,
            |s| s.physical_slot_payload_copy_us = u64::MAX,
            |s| s.physical_install_completions = u64::MAX,
            |s| s.timing_accounting_errors = 1,
            |s| s.evidence_accounting_errors = 1,
            |s| s.physical_slot_prepare_us = 1,
        ] {
            let mut bad = s.clone();
            corrupt(&mut bad);
            assert!(!production_totals(&bad).accounting_valid);
        }
    }

    fn fixture(copy_ratio: f64, scaling: bool) -> Standalone {
        let case = |width: usize, staging: bool| {
            let copy = (1_000_000.0 * width as f64 / if staging { copy_ratio } else { 1.0 }) as u64;
            let wall = if scaling {
                2_000_000
            } else {
                2_000_000 * width as u64
            };
            let batches = (0..20)
                .map(|iteration| Batch {
                    iteration,
                    operations: width as u64,
                    payload_bytes: PAYLOAD as u64 * width as u64,
                    sum_individual_copy_ns: copy,
                    sum_individual_copy_us: copy as f64 / 1000.0,
                    validation_us: 0.0,
                    epoch_write_us: 0.0,
                    wall_ns: wall,
                    wall_us: wall as f64 / 1000.0,
                    view_acquisition_us: 0.0,
                    view_drop_scheduling_us: 0.0,
                    submit_and_drain_us: 0.0,
                    max_simultaneous_jobs_observed: width,
                    failures: vec![],
                    accounting_errors: vec![],
                })
                .collect();
            summarize(
                if staging {
                    "wgpu-staging-view"
                } else {
                    "plain-ram"
                },
                width,
                batches,
                None,
            )
            .unwrap()
        };
        let ram: Vec<_> = WIDTHS.into_iter().map(|w| case(w, false)).collect();
        let wgpu_staging: Vec<_> = WIDTHS.into_iter().map(|w| case(w, true)).collect();
        let comparisons = ram
            .iter()
            .zip(&wgpu_staging)
            .map(|(r, w)| comparison_for(r, w))
            .collect();
        Standalone {
            adapter: AdapterIdentity {
                name: "synthetic".into(),
                vendor: 0,
                device: 0,
                device_type: "synthetic".into(),
                backend: "synthetic".into(),
                driver: "".into(),
                driver_info: "".into(),
            },
            authoritative_l4: false,
            rayon_num_threads: 8,
            max_pending_staging_writes: 8,
            max_pending_staging_bytes: STRIDE * 8,
            source_pattern: "synthetic",
            source_payload_sha256: "".into(),
            warmup_iterations_per_width_and_destination: 3,
            measured_iterations_per_width_and_destination: 20,
            order: "synthetic",
            wall_definition: "synthetic",
            ram,
            wgpu_staging,
            comparisons,
        }
    }

    #[test]
    fn payload_copy_classification_is_material_conservative_and_failure_sensitive() {
        assert_eq!(
            classify(&fixture(0.8, false), None),
            "staging-view-memory-slower-than-ram"
        );
        assert_eq!(
            classify(&fixture(0.95, false), None),
            "no-clear-discriminator"
        );
        assert_eq!(
            classify(&fixture(1.0, true), None),
            "concurrency-amortization-dominant"
        );
        let totals = production_totals(&production_fixture());
        assert_eq!(
            classify(&fixture(1.0, false), Some(&totals)),
            "host-memcpy-bandwidth-dominant"
        );
        let mut failed = fixture(0.8, false);
        failed.wgpu_staging[0].failures.push("injected".into());
        assert_eq!(classify(&failed, Some(&totals)), "no-clear-discriminator");
        let mut noisy = fixture(0.8, false);
        for c in &mut noisy.comparisons {
            c.paired_ratio_p90 = Some(1.01);
        }
        assert_eq!(classify(&noisy, None), "no-clear-discriminator");
    }
}
