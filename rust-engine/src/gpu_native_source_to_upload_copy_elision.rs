//! Standalone diagnostic: full expert pread into an aligned pool allocation or
//! directly into a mapped MAP_WRITE | COPY_SRC allocation. No inference runtime.
//! Hashes/readback/fd evidence are excluded from transfer-cycle and source timers.
use crate::buffer_pool::{BufferPool, PooledBuffer};
use crate::config::Config;
use crate::inference::WeightDtype;
use crate::io_provider::{NvmeStorage, StorageConfig};
use crate::tensor_header::{TensorHeader, UthDtypeId};
use futures::FutureExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

const SCHEMA: &str = "mer.gpu-native-source-to-upload-copy-elision-feasibility.v1";
const FULL: usize = 2_658_304;
const ALIGN: usize = 4096;
const PREFIX: usize = 4096;
const PAYLOAD: usize = 2_654_208;
const EPOCH_OFFSET: usize = 4;
const SLOT: usize = 2_654_212;
const UPLOAD: usize = FULL + ALIGN;
const NAMESPACE: u32 = 48 * 128;
const EPOCH: u32 = 0x1234_5678;
const MAX_ITERATIONS: usize = 65_536;
const GPU_TIMEOUT: Duration = Duration::from_secs(30);

type Result<T> = std::result::Result<T, Failure>;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Args {
    pub(crate) config: PathBuf,
    pub(crate) expected_adapter_name: String,
    pub(crate) warmup_iterations: usize,
    pub(crate) iterations: usize,
    pub(crate) report_out: PathBuf,
}

#[derive(Debug)]
struct Failure {
    classification: &'static str,
    detail: String,
    complete: bool,
}
impl Failure {
    fn runtime(classification: &'static str, detail: impl ToString) -> Self {
        Self {
            classification,
            detail: detail.to_string(),
            complete: false,
        }
    }
    fn authority(detail: impl ToString) -> Self {
        Self {
            classification: "authority-failed",
            detail: detail.to_string(),
            complete: true,
        }
    }
    fn accounting(detail: impl ToString) -> Self {
        Self::runtime("accounting-failed", detail)
    }
}

fn add(dst: &mut u64, n: u64) -> Result<()> {
    *dst = dst
        .checked_add(n)
        .ok_or_else(|| Failure::accounting("counter overflow"))?;
    Ok(())
}
fn elapsed(start: Instant) -> Result<u64> {
    u64::try_from(start.elapsed().as_nanos()).map_err(Failure::accounting)
}
fn timed(dst: &mut u64, start: Instant) -> Result<()> {
    add(dst, elapsed(start)?)
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn finish_sha(hash: &Sha256) -> String {
    format!("{:x}", hash.clone().finalize())
}
fn gbps(bytes: u64, ns: u64) -> Option<f64> {
    (ns > 0).then(|| bytes as f64 / ns as f64)
}

/// Address and offset arithmetic is checked independently of real GPU memory.
/// A page-aligned CPU address alone does not prove GPU copy-offset alignment.
fn aligned_subrange(
    base: usize,
    capacity: usize,
    len: usize,
    align: usize,
) -> std::result::Result<usize, String> {
    if base == 0 || !align.is_power_of_two() || len == 0 || len % align != 0 {
        return Err("invalid direct-read address, alignment, or length".into());
    }
    let offset = (align - base % align) % align;
    let pointer = base.checked_add(offset).ok_or("aligned address overflow")?;
    let end = offset.checked_add(len).ok_or("aligned range overflow")?;
    base.checked_add(end)
        .ok_or("mapped address range overflow")?;
    if pointer % align != 0 || end > capacity {
        return Err("aligned direct-read range is outside mapping".into());
    }
    Ok(offset)
}
fn copy_offsets(
    offset: usize,
    prefix: usize,
    payload: usize,
    capacity: usize,
) -> std::result::Result<u64, String> {
    let source = offset
        .checked_add(prefix)
        .ok_or("GPU source offset overflow")?;
    let end = source
        .checked_add(payload)
        .ok_or("GPU source end overflow")?;
    let dest_end = EPOCH_OFFSET
        .checked_add(payload)
        .ok_or("GPU destination end overflow")?;
    let alignment = wgpu::COPY_BUFFER_ALIGNMENT as usize;
    if source % alignment != 0
        || EPOCH_OFFSET % alignment != 0
        || payload % alignment != 0
        || payload == 0
        || end > capacity
        || dest_end > SLOT
    {
        return Err("GPU copy offset, length, or bounds contract unavailable".into());
    }
    u64::try_from(source).map_err(|e| e.to_string())
}
fn validate_geometry(cfg: &Config) -> Result<()> {
    let m = &cfg.model;
    let payload = m
        .d_model
        .checked_mul(m.d_ff)
        .and_then(|n| n.checked_mul(3))
        .filter(|n| n % 32 == 0)
        .and_then(|n| (n / 32).checked_mul(18));
    if (m.d_model, m.d_ff, m.num_layers, m.num_experts, m.top_k) != (2048, 768, 48, 128, 8)
        || m.dtype != WeightDtype::Q4_0
        || payload != Some(PAYLOAD)
        || m.expert_size != FULL
        || cfg.storage.block_align != ALIGN
        || PREFIX.checked_add(PAYLOAD) != Some(FULL)
        || EPOCH_OFFSET.checked_add(PAYLOAD) != Some(SLOT)
        || FULL % ALIGN != 0
    {
        return Err(Failure::authority("requires exact Qwen3-Coder Q4 geometry: 48x128 experts, 2048x768, full file 2658304, UTH prefix 4096, payload 2654208, slot 2654212"));
    }
    Ok(())
}
fn payload_range(source: &[u8]) -> std::result::Result<(usize, &[u8]), String> {
    let (header, payload) = TensorHeader::strip(source, ALIGN);
    let h = header.ok_or("missing or invalid UTH1 header")?;
    let prefix = source
        .len()
        .checked_sub(payload.len())
        .ok_or("payload offset underflow")?;
    if source.len() != FULL
        || prefix != PREFIX
        || payload.len() != PAYLOAD
        || h.dtype != UthDtypeId::Q4_0
        || h.shape_rank != 3
        || h.shape != [768, 2048, 3, 0]
        || h.quant_scale_count != 0
        || h.quant_scale_offset != 0
    {
        return Err("UTH/payload does not match authoritative full-file Q4 geometry".into());
    }
    Ok((prefix, payload))
}

/// Evenly spaced inclusive namespace samples; after one full namespace, repeat.
/// Encoding for the witness is concatenated u32 little-endian IDs, no framing.
fn expert_sequence(count: usize, total: u32) -> std::result::Result<Vec<u32>, String> {
    if total < 2 || count > MAX_ITERATIONS {
        return Err("invalid sequence bounds".into());
    }
    let span = count.min(total as usize);
    Ok((0..count)
        .map(|i| {
            if span == 1 {
                total / 2
            } else {
                ((i % span) as u64 * (total - 1) as u64 / (span - 1) as u64) as u32
            }
        })
        .collect())
}
fn sequence_sha(ids: &[u32]) -> String {
    let mut hash = Sha256::new();
    for id in ids {
        hash.update(id.to_le_bytes());
    }
    finish_sha(&hash)
}

#[derive(Default, Debug, Serialize)]
struct Times {
    map_wait_ns: u64,
    source_direct_read_ns: u64,
    alignment_setup_ns: u64,
    control_view_acquisition_ns: u64,
    control_cpu_payload_copy_ns: u64,
    control_staging_drop_scheduling_ns: u64,
    treatment_unmap_ns: u64,
    treatment_gpu_copy_encoding_ns: u64,
    epoch_write_ns: u64,
    submit_drain_ns: u64,
    transfer_cycle_ns: u64,
    verification_readback_ns: u64,
}
#[derive(Default, Debug, Serialize)]
struct PointerEvidence {
    observations: u64,
    aligned: u64,
    invalid: u64,
    mapping_base_mod_4096_counts: BTreeMap<usize, u64>,
    aligned_offset_min: Option<usize>,
    aligned_offset_max: Option<usize>,
    gpu_offset_checks: u64,
    gpu_offset_failures: u64,
}
impl PointerEvidence {
    fn observe(&mut self, base: usize, offset: usize, len: usize) -> Result<()> {
        add(&mut self.observations, 1)?;
        if (base + offset) % ALIGN == 0 && len == FULL && len % ALIGN == 0 {
            add(&mut self.aligned, 1)?;
        } else {
            add(&mut self.invalid, 1)?;
        }
        add(
            self.mapping_base_mod_4096_counts
                .entry(base % ALIGN)
                .or_default(),
            1,
        )?;
        self.aligned_offset_min = Some(self.aligned_offset_min.map_or(offset, |v| v.min(offset)));
        self.aligned_offset_max = Some(self.aligned_offset_max.map_or(offset, |v| v.max(offset)));
        Ok(())
    }
}
#[derive(Default, Debug, Serialize)]
struct FdEvidence {
    checks: u64,
    direct_observed: u64,
    full_file_length_observed: u64,
    flags_counts: BTreeMap<i32, u64>,
    failures: u64,
}
#[derive(Default, Debug, Serialize)]
struct Arm {
    ops_attempted: u64,
    source_read_attempts: u64,
    source_read_ops: u64,
    full_source_bytes: u64,
    payload_ops: u64,
    payload_bytes: u64,
    cpu_payload_copy_bytes: u64,
    upload_ops: u64,
    gpu_copied_bytes: u64,
    explicit_copy_buffer_bytes: u64,
    epoch_bytes: u64,
    gpu_completed_ops: u64,
    verified_ops: u64,
    verification_readback_bytes: u64,
    verification_destination_reset_ops: u64,
    verification_destination_reset_bytes: u64,
    map_attempts: u64,
    maps_completed: u64,
    unmaps: u64,
    alignment_failures: u64,
    mapped_direct_io_rejections: u64,
    rejection_errno_counts: BTreeMap<i32, u64>,
    map_failures: u64,
    source_failures: u64,
    gpu_failures: u64,
    accounting_failures: u64,
    exact_read_length_failures: u64,
    fallback_reads: u64,
    pointers: PointerEvidence,
    fd_evidence: FdEvidence,
    times: Times,
    source_gbps: Option<f64>,
    cpu_payload_copy_gbps: Option<f64>,
    submit_drain_payload_gbps: Option<f64>,
    transfer_cycle_payload_gbps: Option<f64>,
}
impl Arm {
    fn rates(&mut self) {
        self.source_gbps = gbps(self.full_source_bytes, self.times.source_direct_read_ns);
        self.cpu_payload_copy_gbps = gbps(
            self.cpu_payload_copy_bytes,
            self.times.control_cpu_payload_copy_ns,
        );
        self.submit_drain_payload_gbps = gbps(self.gpu_copied_bytes, self.times.submit_drain_ns);
        self.transfer_cycle_payload_gbps =
            gbps(self.gpu_copied_bytes, self.times.transfer_cycle_ns);
    }
    fn reconcile(&self, treatment: bool) -> bool {
        let expected = |ops: u64, bytes: usize| ops.checked_mul(bytes as u64);
        expected(self.source_read_ops, FULL) == Some(self.full_source_bytes)
            && expected(self.payload_ops, PAYLOAD) == Some(self.payload_bytes)
            && expected(self.upload_ops, PAYLOAD) == Some(self.gpu_copied_bytes)
            && expected(self.upload_ops, EPOCH_OFFSET) == Some(self.epoch_bytes)
            && expected(self.verified_ops, SLOT) == Some(self.verification_readback_bytes)
            && expected(self.verification_destination_reset_ops, SLOT)
                == Some(self.verification_destination_reset_bytes)
            && self.cpu_payload_copy_bytes == if treatment { 0 } else { self.gpu_copied_bytes }
            && self.explicit_copy_buffer_bytes == if treatment { self.gpu_copied_bytes } else { 0 }
            && self.source_read_ops <= self.source_read_attempts
            && self.source_read_attempts <= self.ops_attempted
            && self.payload_ops <= self.source_read_ops
            && self.upload_ops <= self.payload_ops
            && self.gpu_completed_ops <= self.upload_ops
            && self.verified_ops <= self.gpu_completed_ops
            && self.pointers.aligned == self.source_read_attempts
            && self.pointers.invalid == 0
            && self.exact_read_length_failures == 0
            && self.fallback_reads == 0
            && self.accounting_failures == 0
    }
    fn success(&self, expected: u64, treatment: bool) -> bool {
        self.reconcile(treatment)
            && self.ops_attempted == expected
            && self.verified_ops == expected
            && self.fd_evidence.checks == expected
            && self.fd_evidence.direct_observed == expected
            && self.fd_evidence.full_file_length_observed == expected
            && self.fd_evidence.failures == 0
            && self.verification_destination_reset_ops == expected
            && self.source_read_ops == expected
            && self.source_read_attempts == expected
            && self.source_failures == 0
            && self.gpu_failures == 0
            && self.map_failures == 0
            && self.alignment_failures == 0
            && self.mapped_direct_io_rejections == 0
            && (!treatment
                || (self.maps_completed == expected
                    && self.map_attempts == expected
                    && self.unmaps == expected
                    && self.pointers.gpu_offset_checks == expected
                    && self.pointers.gpu_offset_failures == 0))
    }
}

#[derive(Default)]
struct Streams {
    source: Sha256,
    payload: Sha256,
    gpu: Sha256,
}
#[derive(Default, Debug, Serialize)]
struct Witnesses {
    full_source_sha256: String,
    bare_payload_sha256: String,
    gpu_destination_payload_sha256: String,
}
impl Streams {
    fn snapshot(&self) -> Witnesses {
        Witnesses {
            full_source_sha256: finish_sha(&self.source),
            bare_payload_sha256: finish_sha(&self.payload),
            gpu_destination_payload_sha256: finish_sha(&self.gpu),
        }
    }
    fn source(&mut self, bytes: &[u8]) -> std::result::Result<(usize, Hashes), String> {
        let (offset, payload) = payload_range(bytes)?;
        self.source.update(bytes);
        self.payload.update(payload);
        Ok((
            offset,
            Hashes {
                source: sha(bytes),
                payload: sha(payload),
                gpu: String::new(),
                epoch: false,
            },
        ))
    }
}
#[derive(Debug)]
struct Hashes {
    source: String,
    payload: String,
    gpu: String,
    epoch: bool,
}
#[derive(Debug)]
enum Outcome {
    Verified(Hashes),
    MappedRejected,
    AlignmentUnavailable,
}
#[derive(Debug, Serialize)]
struct Mismatch {
    phase: &'static str,
    pair: usize,
    expert_id: u32,
    kind: &'static str,
    control: String,
    treatment: String,
}
#[derive(Debug, Serialize)]
struct Phase {
    name: &'static str,
    ordered_expert_ids: Vec<u32>,
    expert_id_sequence_sha256: String,
    attempted_expert_id_sequence_sha256: String,
    pairs_attempted: u64,
    pairs_completed: u64,
    control: Arm,
    treatment: Arm,
    control_witnesses: Witnesses,
    treatment_witnesses: Witnesses,
    mismatch_count: u64,
    first_mismatch: Option<Mismatch>,
    first_mechanism_rejection: Option<String>,
}
impl Phase {
    fn new(name: &'static str, ids: Vec<u32>) -> Self {
        Self {
            name,
            expert_id_sequence_sha256: sequence_sha(&ids),
            ordered_expert_ids: ids,
            attempted_expert_id_sequence_sha256: sequence_sha(&[]),
            pairs_attempted: 0,
            pairs_completed: 0,
            control: Arm::default(),
            treatment: Arm::default(),
            control_witnesses: Witnesses::default(),
            treatment_witnesses: Witnesses::default(),
            mismatch_count: 0,
            first_mismatch: None,
            first_mechanism_rejection: None,
        }
    }
    fn mismatch(
        &mut self,
        pair: usize,
        id: u32,
        kind: &'static str,
        control: &str,
        treatment: &str,
    ) -> Result<()> {
        if control != treatment {
            add(&mut self.mismatch_count, 1)?;
            self.first_mismatch.get_or_insert_with(|| Mismatch {
                phase: self.name,
                pair,
                expert_id: id,
                kind,
                control: control.into(),
                treatment: treatment.into(),
            });
        }
        Ok(())
    }
    fn compare(
        &mut self,
        pair: usize,
        id: u32,
        control: &Outcome,
        treatment: &Outcome,
    ) -> Result<()> {
        if let Outcome::Verified(c) = control {
            self.mismatch(pair, id, "control-gpu-payload", &c.payload, &c.gpu)?;
            self.mismatch(
                pair,
                id,
                "control-epoch",
                "true",
                if c.epoch { "true" } else { "false" },
            )?;
        }
        if let Outcome::Verified(t) = treatment {
            self.mismatch(pair, id, "treatment-gpu-payload", &t.payload, &t.gpu)?;
            self.mismatch(
                pair,
                id,
                "treatment-epoch",
                "true",
                if t.epoch { "true" } else { "false" },
            )?;
        }
        if let (Outcome::Verified(c), Outcome::Verified(t)) = (control, treatment) {
            self.mismatch(pair, id, "full-source", &c.source, &t.source)?;
            self.mismatch(pair, id, "bare-payload", &c.payload, &t.payload)?;
            self.mismatch(pair, id, "gpu-destination-payload", &c.gpu, &t.gpu)?;
        }
        Ok(())
    }
    fn parity(&self) -> bool {
        let c = &self.control_witnesses;
        let t = &self.treatment_witnesses;
        self.mismatch_count == 0
            && c.full_source_sha256 == t.full_source_sha256
            && c.bare_payload_sha256 == t.bare_payload_sha256
            && c.bare_payload_sha256 == c.gpu_destination_payload_sha256
            && c.bare_payload_sha256 == t.gpu_destination_payload_sha256
    }
    fn accounted(&self) -> bool {
        self.pairs_completed == self.ordered_expert_ids.len() as u64
            && self.pairs_attempted == self.pairs_completed
            && self.attempted_expert_id_sequence_sha256 == self.expert_id_sequence_sha256
            && self.control.reconcile(false)
            && self.treatment.reconcile(true)
    }
    fn successful(&self) -> bool {
        let n = self.ordered_expert_ids.len() as u64;
        self.accounted()
            && self.control.success(n, false)
            && self.treatment.success(n, true)
            && self.parity()
    }
}

#[derive(Default, Debug, Serialize)]
struct Authority {
    linux: bool,
    expected_adapter_name: String,
    adapter_name: Option<String>,
    adapter_backend: Option<String>,
    adapter_device_type: Option<String>,
    adapter_vendor: Option<u32>,
    adapter_device: Option<u32>,
    driver: Option<String>,
    driver_info: Option<String>,
    adapter_authoritative: bool,
    direct_io_requested: bool,
    packed_storage: Option<bool>,
    source_data_dir: Option<PathBuf>,
    exact_geometry: bool,
    full_source_bytes: usize,
    block_alignment: usize,
    uth_prefix_bytes: usize,
    bare_payload_bytes: usize,
    physical_slot_bytes: usize,
    upload_capacity_bytes: usize,
}
#[derive(Debug, Serialize)]
struct Report {
    schema: &'static str,
    args: Args,
    config_sha256: Option<String>,
    complete: bool,
    feasibility_pass: bool,
    classification: String,
    failure: Option<String>,
    runtime_failures: u64,
    accounting_failures: u64,
    authority: Authority,
    warmup: Phase,
    measured: Phase,
    source_throughput_ratio_treatment_over_control: Option<f64>,
    source_throughput_minimum_ratio: f64,
    total_cycle_superiority_required: bool,
    sequence_contract: &'static str,
    timing_contract: &'static str,
    odirect_evidence_contract: &'static str,
    cpu_copy_accounting_contract: &'static str,
}
impl Report {
    fn new(args: Args) -> Self {
        Self { schema: SCHEMA, authority: Authority {
                linux: cfg!(target_os = "linux"), expected_adapter_name: args.expected_adapter_name.clone(),
                full_source_bytes: FULL, block_alignment: ALIGN, uth_prefix_bytes: PREFIX,
                bare_payload_bytes: PAYLOAD, physical_slot_bytes: SLOT, upload_capacity_bytes: UPLOAD,
                ..Authority::default()
            }, args, config_sha256: None, complete: false, feasibility_pass: false,
            classification: "not-completed".into(), failure: None, runtime_failures: 0, accounting_failures: 0,
            warmup: Phase::new("warmup", vec![]), measured: Phase::new("measured", vec![]),
            source_throughput_ratio_treatment_over_control: None, source_throughput_minimum_ratio: 0.9,
            total_cycle_superiority_required: false,
            sequence_contract: "global ID = layer*128+local; span=min(count,6144); ID[i]=floor((i%span)*6143/(span-1)); singleton warmup=3072; SHA256(concatenated u32 LE IDs). Even pair CONTROL,TREATMENT; odd pair TREATMENT,CONTROL; parity hashes concatenate exact source/payload bytes in pair order, without framing. Warmup is separate and excluded.",
            timing_contract: "nanoseconds, decimal GB/s=bytes/ns. Source timer covers only read_expert/read_expert_into_aligned_slice (cached fd lookup, block_in_place, pread and unchanged retries/breakers). Each arm's total cycle is its wall time minus fd evidence, source/header hashing and GPU readback verification. Each standalone destination is GPU-cleared before its arm, outside transfer timers, to prevent stale paired data from satisfying parity. No verification enters the 90% source-throughput gate. submit/drain includes queued epoch and payload; it is descriptive, not isolated DMA bandwidth.",
            odirect_evidence_contract: "Before each arm: Linux fcntl(F_GETFL) on the actual cached expert fd plus fstat file length. Exclusively owned sequential storage retains that same fd through the read; no packed storage, no fallback reads, no dense tensors or inference.",
            cpu_copy_accounting_contract: "CONTROL counts exact bytes passed to QueueWriteBufferView::copy_from_slice; TREATMENT reads the entire source file directly into BufferViewMut, performs no CPU payload copy, then unmaps and encodes the bare payload GPU copy. Hashing/readback are excluded verification work. gpu_copied_bytes excludes the separately counted 4-byte epoch and readback bytes.",
        }
    }
    fn authoritative(&self) -> bool {
        let a = &self.authority;
        a.linux
            && a.expected_adapter_name == "NVIDIA L4"
            && a.adapter_authoritative
            && a.direct_io_requested
            && a.packed_storage == Some(false)
            && a.exact_geometry
    }
    fn fail(&mut self, failure: Failure) {
        self.complete = failure.complete;
        self.feasibility_pass = false;
        self.classification = failure.classification.into();
        self.failure = Some(failure.detail);
        if !failure.complete {
            self.runtime_failures += 1;
        }
        if failure.classification == "accounting-failed" {
            self.accounting_failures += 1;
        }
    }
    fn classify(&mut self) {
        self.warmup.control.rates();
        self.warmup.treatment.rates();
        self.measured.control.rates();
        self.measured.treatment.rates();
        self.complete = true;
        self.feasibility_pass = false;
        if !self.authoritative() {
            self.classification = "authority-failed".into();
            return;
        }
        if !self.warmup.accounted() || !self.measured.accounted() || self.accounting_failures != 0 {
            self.fail(Failure::accounting(
                "phase/byte/operation/sequence reconciliation failed",
            ));
            return;
        }
        let phases = [&self.warmup, &self.measured];
        if phases.iter().any(|p| p.mismatch_count > 0) {
            self.classification = if phases.iter().any(|p| {
                p.first_mismatch
                    .as_ref()
                    .is_some_and(|m| matches!(m.kind, "full-source" | "bare-payload"))
            }) {
                "source-parity-failed"
            } else {
                "gpu-copy-parity-failed"
            }
            .into();
            return;
        }
        let controls_ok = phases
            .iter()
            .all(|p| p.control.success(p.ordered_expert_ids.len() as u64, false));
        let rejected = phases.iter().all(|p| {
            let t = &p.treatment;
            let n = p.ordered_expert_ids.len() as u64;
            t.ops_attempted == n
                && t.source_read_attempts == n
                && t.mapped_direct_io_rejections == n
                && t.source_failures == n
                && t.source_read_ops == 0
                && t.upload_ops == 0
                && t.map_attempts == n
                && t.maps_completed == n
                && t.unmaps == n
                && t.verification_destination_reset_ops == n
                && t.fd_evidence.direct_observed == n
                && t.fd_evidence.full_file_length_observed == n
                && t.map_failures == 0
                && t.gpu_failures == 0
                && t.alignment_failures == 0
                && t.rejection_errno_counts.values().sum::<u64>() == n
                && t.rejection_errno_counts
                    .keys()
                    .all(|e| mapped_rejection(Some(*e)))
        });
        if controls_ok && rejected && self.measured.treatment.mapped_direct_io_rejections >= 2 {
            self.classification = "mapped-upload-direct-io-rejected".into();
            return;
        }
        if controls_ok
            && phases.iter().any(|p| p.treatment.alignment_failures > 0)
            && phases.iter().all(|p| {
                p.treatment.source_failures == 0
                    && p.treatment.map_failures == 0
                    && p.treatment.gpu_failures == 0
            })
        {
            self.classification = "alignment-contract-unavailable".into();
            return;
        }
        if !self.warmup.successful() || !self.measured.successful() || self.runtime_failures != 0 {
            self.fail(Failure::runtime("runtime-failed", "incomplete or inconsistent arm evidence; mapped I/O rejection requires every attempted treatment read to reject and every CONTROL to work"));
            return;
        }
        let c = &self.measured.control;
        let t = &self.measured.treatment;
        let ratio = match (c.source_gbps, t.source_gbps) {
            (Some(c), Some(t)) if c > 0.0 && c.is_finite() && t.is_finite() => t / c,
            _ => {
                self.fail(Failure::accounting(
                    "missing or invalid source-only throughput",
                ));
                return;
            }
        };
        self.source_throughput_ratio_treatment_over_control = Some(ratio);
        // Successful arms have identical source bytes, so this is the exact
        // rational 90% gate, without a floating-point boundary tolerance.
        self.feasibility_pass = u128::from(c.times.source_direct_read_ns) * 10
            >= u128::from(t.times.source_direct_read_ns) * 9;
        self.classification = if self.feasibility_pass {
            "copy-elision-feasible"
        } else {
            "mapped-source-too-slow"
        }
        .into();
    }
}
fn mapped_rejection(errno: Option<i32>) -> bool {
    matches!(errno, Some(libc::EINVAL) | Some(libc::EFAULT))
}

/// All buffers and callbacks belong exclusively to this diagnostic.
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    upload: wgpu::Buffer,
    destination: wgpu::Buffer,
    readback: wgpu::Buffer,
    errors: Arc<Mutex<Option<String>>>,
}
impl Gpu {
    async fn new(a: &mut Authority) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .into_iter()
            .find(|adapter| {
                let info = adapter.get_info();
                info.name == "NVIDIA L4"
                    && info.backend == wgpu::Backend::Vulkan
                    && info.device_type == wgpu::DeviceType::DiscreteGpu
            })
            .ok_or_else(|| {
                Failure::authority("exact discrete NVIDIA L4 Vulkan adapter unavailable")
            })?;
        let info = adapter.get_info();
        a.adapter_name = Some(info.name.clone());
        a.adapter_backend = Some(format!("{:?}", info.backend));
        a.adapter_device_type = Some(format!("{:?}", info.device_type));
        a.adapter_vendor = Some(info.vendor);
        a.adapter_device = Some(info.device);
        a.driver = Some(info.driver);
        a.driver_info = Some(info.driver_info);
        a.adapter_authoritative = a.linux
            && info.name == a.expected_adapter_name
            && info.name == "NVIDIA L4"
            && info.backend == wgpu::Backend::Vulkan
            && info.device_type == wgpu::DeviceType::DiscreteGpu;
        if !a.adapter_authoritative {
            return Err(Failure::authority("adapter authority mismatch"));
        }
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("source-to-upload-diagnostic"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .map_err(|e| Failure::runtime("gpu-failed", e))?;
        let errors = Arc::new(Mutex::new(None));
        let uncaptured = errors.clone();
        device.on_uncaptured_error(Box::new(move |error| {
            uncaptured
                .lock()
                .unwrap()
                .get_or_insert_with(|| format!("wgpu: {error}"));
        }));
        let lost = errors.clone();
        device.set_device_lost_callback(move |reason, message| {
            lost.lock()
                .unwrap()
                .get_or_insert_with(|| format!("device lost {reason:?}: {message}"));
        });
        let upload = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mapped-direct-source-upload"),
            size: UPLOAD as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let destination = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("standalone-physical-slot"),
            size: SLOT as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verification-only-slot-readback"),
            size: SLOT as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gpu = Self {
            device,
            queue,
            upload,
            destination,
            readback,
            errors,
        };
        gpu.check()?;
        Ok(gpu)
    }
    fn check(&self) -> Result<()> {
        match self
            .errors
            .lock()
            .map_err(|e| Failure::runtime("gpu-failed", e))?
            .as_ref()
        {
            Some(e) => Err(Failure::runtime("gpu-failed", e)),
            None => Ok(()),
        }
    }
    fn wait<T>(&self, rx: &mpsc::Receiver<T>) -> Result<T> {
        let start = Instant::now();
        loop {
            self.device.poll(wgpu::Maintain::Poll);
            self.check()?;
            match rx.recv_timeout(Duration::from_millis(1)) {
                Ok(value) => return Ok(value),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(Failure::runtime(
                        "gpu-failed",
                        "GPU callback channel disconnected",
                    ))
                }
                Err(mpsc::RecvTimeoutError::Timeout) if start.elapsed() >= GPU_TIMEOUT => {
                    return Err(Failure::runtime(
                        "gpu-failed",
                        "GPU callback timed out after 30 seconds",
                    ))
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }
    fn map(&self, buffer: &wgpu::Buffer, mode: wgpu::MapMode) -> Result<()> {
        let (tx, rx) = mpsc::channel();
        buffer.slice(..).map_async(mode, move |r| {
            let _ = tx.send(r);
        });
        match self
            .wait(&rx)
            .and_then(|r| r.map_err(|e| Failure::runtime("map-failed", e)))
        {
            Ok(()) => Ok(()),
            Err(e) => {
                buffer.unmap();
                Err(e)
            }
        }
    }
    fn drain(&self, command: Option<wgpu::CommandBuffer>) -> Result<()> {
        self.queue.submit(command);
        let (tx, rx) = mpsc::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = tx.send(());
        });
        self.wait(&rx)
    }
    fn verify(&self, hashes: &mut Hashes, stream: &mut Streams) -> Result<()> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("verification-only-readback"),
            });
        encoder.copy_buffer_to_buffer(&self.destination, 0, &self.readback, 0, SLOT as u64);
        self.drain(Some(encoder.finish()))?;
        self.map(&self.readback, wgpu::MapMode::Read)?;
        {
            let view = self.readback.slice(..).get_mapped_range();
            hashes.epoch = view[..EPOCH_OFFSET] == EPOCH.to_le_bytes();
            hashes.gpu = sha(&view[EPOCH_OFFSET..]);
            stream.gpu.update(&view[EPOCH_OFFSET..]);
        }
        self.readback.unmap();
        self.check()
    }

    fn reset_destination(&self) -> Result<()> {
        // Verification preparation prevents an even pair's TREATMENT from
        // passing via the preceding CONTROL payload left in the shared slot.
        // This private GPU clear is outside transfer timers and CPU-copy bytes.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("verification-only-destination-reset"),
            });
        encoder.clear_buffer(&self.destination, 0, None);
        self.drain(Some(encoder.finish()))
    }
}

fn prepare_destination(gpu: &Gpu, arm: &mut Arm) -> Result<()> {
    let start = Instant::now();
    let result = gpu.reset_destination();
    timed(&mut arm.times.verification_readback_ns, start)?;
    if result.is_err() {
        add(&mut arm.gpu_failures, 1)?;
    }
    result?;
    add(&mut arm.verification_destination_reset_ops, 1)?;
    add(&mut arm.verification_destination_reset_bytes, SLOT as u64)
}

fn fd_evidence(storage: &NvmeStorage, id: u32, arm: &mut Arm) -> Result<()> {
    let start = Instant::now();
    let evidence = storage.source_to_upload_fd_evidence(id);
    timed(&mut arm.times.verification_readback_ns, start)?;
    add(&mut arm.fd_evidence.checks, 1)?;
    let (flags, direct, len) = evidence.map_err(|e| {
        arm.fd_evidence.failures += 1;
        Failure::runtime("source-failed", format!("fd evidence for expert {id}: {e}"))
    })?;
    add(arm.fd_evidence.flags_counts.entry(flags).or_default(), 1)?;
    if direct {
        add(&mut arm.fd_evidence.direct_observed, 1)?;
    }
    if len == FULL as u64 {
        add(&mut arm.fd_evidence.full_file_length_observed, 1)?;
    }
    if !direct || len != FULL as u64 {
        return Err(Failure::authority(format!(
            "expert {id}: actual fd O_DIRECT={direct}, full file length={len}, flags={flags}"
        )));
    }
    Ok(())
}
fn read_result(
    result: io::Result<usize>,
    treatment: bool,
    arm: &mut Arm,
    id: u32,
    rejection: &mut Option<String>,
) -> Result<bool> {
    match result {
        Ok(n) if n == FULL => {
            add(&mut arm.source_read_ops, 1)?;
            add(&mut arm.full_source_bytes, n as u64)?;
            Ok(true)
        }
        Ok(n) => {
            add(&mut arm.exact_read_length_failures, 1)?;
            Err(Failure::accounting(format!(
                "expert {id}: read length {n}, expected {FULL}"
            )))
        }
        Err(e) => {
            add(&mut arm.source_failures, 1)?;
            if e.kind() == io::ErrorKind::UnexpectedEof {
                add(&mut arm.exact_read_length_failures, 1)?;
            }
            if treatment && mapped_rejection(e.raw_os_error()) {
                add(&mut arm.mapped_direct_io_rejections, 1)?;
                add(
                    arm.rejection_errno_counts
                        .entry(e.raw_os_error().unwrap())
                        .or_default(),
                    1,
                )?;
                rejection.get_or_insert_with(|| {
                    format!(
                        "expert {id}: mapped full-file pread rejected: {e}; errno={:?}",
                        e.raw_os_error()
                    )
                });
                Ok(false)
            } else {
                Err(Failure::runtime(
                    "source-failed",
                    format!("expert {id}: {e}; errno={:?}", e.raw_os_error()),
                ))
            }
        }
    }
}
fn note_payload(arm: &mut Arm) -> Result<()> {
    add(&mut arm.payload_ops, 1)?;
    add(&mut arm.payload_bytes, PAYLOAD as u64)
}
fn note_upload(arm: &mut Arm, treatment: bool) -> Result<()> {
    add(&mut arm.upload_ops, 1)?;
    add(&mut arm.gpu_copied_bytes, PAYLOAD as u64)?;
    add(&mut arm.epoch_bytes, EPOCH_OFFSET as u64)?;
    if treatment {
        add(&mut arm.explicit_copy_buffer_bytes, PAYLOAD as u64)?;
    }
    Ok(())
}
fn verify(gpu: &Gpu, arm: &mut Arm, hashes: &mut Hashes, streams: &mut Streams) -> Result<()> {
    let start = Instant::now();
    let result = gpu.verify(hashes, streams);
    timed(&mut arm.times.verification_readback_ns, start)?;
    if result.is_err() {
        add(&mut arm.gpu_failures, 1)?;
    }
    result?;
    add(&mut arm.verified_ops, 1)?;
    add(&mut arm.verification_readback_bytes, SLOT as u64)
}

async fn control(
    gpu: &Gpu,
    storage: &NvmeStorage,
    buf: &mut PooledBuffer,
    id: u32,
    arm: &mut Arm,
    streams: &mut Streams,
) -> Result<Outcome> {
    add(&mut arm.ops_attempted, 1)?;
    fd_evidence(storage, id, arm)?;
    prepare_destination(gpu, arm)?;
    let cycle = Instant::now();
    let verification_before = arm.times.verification_readback_ns;
    let result: Result<Outcome> = async {
        let start = Instant::now();
        let base = buf.as_slice().as_ptr() as usize;
        let offset = aligned_subrange(base, buf.len(), FULL, ALIGN).map_err(Failure::accounting)?;
        if offset != 0 {
            return Err(Failure::accounting(
                "CONTROL pool buffer is not page aligned",
            ));
        }
        arm.pointers.observe(base, 0, buf.len())?;
        timed(&mut arm.times.alignment_setup_ns, start)?;
        add(&mut arm.source_read_attempts, 1)?;
        let start = Instant::now();
        let read = storage.read_expert(id, buf).await;
        timed(&mut arm.times.source_direct_read_ns, start)?;
        read_result(read, false, arm, id, &mut None)?;
        let start = Instant::now();
        let parsed = streams.source(buf.as_slice());
        timed(&mut arm.times.verification_readback_ns, start)?;
        let (offset, mut hashes) =
            parsed.map_err(|e| Failure::authority(format!("expert {id}: {e}")))?;
        note_payload(arm)?;
        let start = Instant::now();
        gpu.queue
            .write_buffer(&gpu.destination, 0, &EPOCH.to_le_bytes());
        timed(&mut arm.times.epoch_write_ns, start)?;
        let start = Instant::now();
        let view = gpu.queue.write_buffer_with(
            &gpu.destination,
            EPOCH_OFFSET as u64,
            NonZeroU64::new(PAYLOAD as u64).unwrap(),
        );
        timed(&mut arm.times.control_view_acquisition_ns, start)?;
        let mut view = view.ok_or_else(|| {
            Failure::runtime("gpu-failed", "Queue::write_buffer_with returned None")
        })?;
        let start = Instant::now();
        view.copy_from_slice(&buf.as_slice()[offset..]);
        timed(&mut arm.times.control_cpu_payload_copy_ns, start)?;
        add(&mut arm.cpu_payload_copy_bytes, PAYLOAD as u64)?;
        let start = Instant::now();
        drop(view);
        timed(&mut arm.times.control_staging_drop_scheduling_ns, start)?;
        gpu.check()?;
        note_upload(arm, false)?;
        let start = Instant::now();
        let result = gpu.drain(None);
        timed(&mut arm.times.submit_drain_ns, start)?;
        result?;
        add(&mut arm.gpu_completed_ops, 1)?;
        verify(gpu, arm, &mut hashes, streams)?;
        Ok(Outcome::Verified(hashes))
    }
    .await;
    let verification = arm
        .times
        .verification_readback_ns
        .checked_sub(verification_before)
        .ok_or_else(|| Failure::accounting("verification timer underflow"))?;
    let transfer = elapsed(cycle)?
        .checked_sub(verification)
        .ok_or_else(|| Failure::accounting("transfer timer underflow"))?;
    add(&mut arm.times.transfer_cycle_ns, transfer)?;
    if result
        .as_ref()
        .is_err_and(|e| e.classification == "gpu-failed")
        && arm.gpu_failures == 0
    {
        add(&mut arm.gpu_failures, 1)?;
    }
    result
}

async fn treatment(
    gpu: &Gpu,
    storage: &NvmeStorage,
    id: u32,
    arm: &mut Arm,
    streams: &mut Streams,
    rejection: &mut Option<String>,
) -> Result<Outcome> {
    add(&mut arm.ops_attempted, 1)?;
    fd_evidence(storage, id, arm)?;
    prepare_destination(gpu, arm)?;
    let cycle = Instant::now();
    let verification_before = arm.times.verification_readback_ns;
    let result: Result<Outcome> = async {
        add(&mut arm.map_attempts, 1)?;
        let start = Instant::now();
        let mapped = gpu.map(&gpu.upload, wgpu::MapMode::Write);
        timed(&mut arm.times.map_wait_ns, start)?;
        if mapped.is_err() {
            add(&mut arm.map_failures, 1)?;
        }
        mapped?;
        add(&mut arm.maps_completed, 1)?;
        // This inner future borrows the view only. It completes and drops every
        // BufferViewMut before the outer code unmaps or submits any GPU work.
        let source = async {
            let start = Instant::now();
            let mut view = gpu.upload.slice(..).get_mapped_range_mut();
            let base = view.as_ptr() as usize;
            // Recomputed for EVERY map, including every warmup and measured op.
            let offset = aligned_subrange(base, view.len(), FULL, ALIGN).and_then(|offset| {
                copy_offsets(offset, PREFIX, PAYLOAD, view.len()).map(|_| offset)
            });
            timed(&mut arm.times.alignment_setup_ns, start)?;
            let offset = match offset {
                Ok(offset) => offset,
                Err(e) => {
                    add(&mut arm.alignment_failures, 1)?;
                    add(
                        arm.pointers
                            .mapping_base_mod_4096_counts
                            .entry(base % ALIGN)
                            .or_default(),
                        1,
                    )?;
                    rejection.get_or_insert_with(|| {
                        format!("expert {id}: {e}; mapped base modulo 4096={}", base % ALIGN)
                    });
                    return Ok(None);
                }
            };
            arm.pointers.observe(base, offset, FULL)?;
            add(&mut arm.source_read_attempts, 1)?;
            let start = Instant::now();
            let read = storage
                .read_expert_into_aligned_slice(id, &mut view[offset..offset + FULL])
                .await;
            timed(&mut arm.times.source_direct_read_ns, start)?;
            if !read_result(read, true, arm, id, rejection)? {
                return Ok(Some(Err(Outcome::MappedRejected)));
            }
            let start = Instant::now();
            let parsed = streams.source(&view[offset..offset + FULL]);
            timed(&mut arm.times.verification_readback_ns, start)?;
            let (prefix, hashes) =
                parsed.map_err(|e| Failure::authority(format!("expert {id}: {e}")))?;
            note_payload(arm)?;
            let start = Instant::now();
            let gpu_offset = copy_offsets(offset, prefix, PAYLOAD, view.len());
            timed(&mut arm.times.alignment_setup_ns, start)?;
            add(&mut arm.pointers.gpu_offset_checks, 1)?;
            let gpu_offset = gpu_offset.map_err(|e| {
                arm.pointers.gpu_offset_failures += 1;
                Failure::accounting(e)
            })?;
            Ok(Some(Ok((gpu_offset, hashes))))
        }
        .await;
        let start = Instant::now();
        gpu.upload.unmap();
        timed(&mut arm.times.treatment_unmap_ns, start)?;
        add(&mut arm.unmaps, 1)?;
        gpu.check()?;
        let (gpu_offset, mut hashes) = match source? {
            None => return Ok(Outcome::AlignmentUnavailable),
            Some(Err(outcome)) => return Ok(outcome),
            Some(Ok(data)) => data,
        };
        let start = Instant::now();
        gpu.queue
            .write_buffer(&gpu.destination, 0, &EPOCH.to_le_bytes());
        timed(&mut arm.times.epoch_write_ns, start)?;
        let start = Instant::now();
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("source-to-upload-payload-copy"),
            });
        encoder.copy_buffer_to_buffer(
            &gpu.upload,
            gpu_offset,
            &gpu.destination,
            EPOCH_OFFSET as u64,
            PAYLOAD as u64,
        );
        let command = encoder.finish();
        timed(&mut arm.times.treatment_gpu_copy_encoding_ns, start)?;
        gpu.check()?;
        note_upload(arm, true)?;
        let start = Instant::now();
        let result = gpu.drain(Some(command));
        timed(&mut arm.times.submit_drain_ns, start)?;
        result?;
        add(&mut arm.gpu_completed_ops, 1)?;
        verify(gpu, arm, &mut hashes, streams)?;
        Ok(Outcome::Verified(hashes))
    }
    .await;
    let verification = arm
        .times
        .verification_readback_ns
        .checked_sub(verification_before)
        .ok_or_else(|| Failure::accounting("verification timer underflow"))?;
    let transfer = elapsed(cycle)?
        .checked_sub(verification)
        .ok_or_else(|| Failure::accounting("transfer timer underflow"))?;
    add(&mut arm.times.transfer_cycle_ns, transfer)?;
    if result
        .as_ref()
        .is_err_and(|e| e.classification == "gpu-failed")
        && arm.gpu_failures == 0
    {
        add(&mut arm.gpu_failures, 1)?;
    }
    result
}

async fn run_phase(
    phase: &mut Phase,
    gpu: &Gpu,
    storage: &NvmeStorage,
    buf: &mut PooledBuffer,
) -> Result<()> {
    let mut control_streams = Streams::default();
    let mut treatment_streams = Streams::default();
    let mut attempted_ids = Sha256::new();
    let result = async {
        for pair in 0..phase.ordered_expert_ids.len() {
            let id = phase.ordered_expert_ids[pair];
            add(&mut phase.pairs_attempted, 1)?;
            attempted_ids.update(id.to_le_bytes());
            let (c, t) = if pair % 2 == 0 {
                let c = control(
                    gpu,
                    storage,
                    buf,
                    id,
                    &mut phase.control,
                    &mut control_streams,
                )
                .await?;
                let t = treatment(
                    gpu,
                    storage,
                    id,
                    &mut phase.treatment,
                    &mut treatment_streams,
                    &mut phase.first_mechanism_rejection,
                )
                .await?;
                (c, t)
            } else {
                let t = treatment(
                    gpu,
                    storage,
                    id,
                    &mut phase.treatment,
                    &mut treatment_streams,
                    &mut phase.first_mechanism_rejection,
                )
                .await?;
                let c = control(
                    gpu,
                    storage,
                    buf,
                    id,
                    &mut phase.control,
                    &mut control_streams,
                )
                .await?;
                (c, t)
            };
            phase.compare(pair, id, &c, &t)?;
            add(&mut phase.pairs_completed, 1)?;
        }
        Ok(())
    }
    .await;
    phase.control_witnesses = control_streams.snapshot();
    phase.treatment_witnesses = treatment_streams.snapshot();
    phase.attempted_expert_id_sequence_sha256 = finish_sha(&attempted_ids);
    phase.control.rates();
    phase.treatment.rates();
    result
}

async fn execute(report: &mut Report) -> Result<()> {
    if !(2..=MAX_ITERATIONS).contains(&report.args.iterations)
        || report.args.warmup_iterations > MAX_ITERATIONS
    {
        return Err(Failure::runtime(
            "invalid-arguments",
            "iterations must be 2..=65536; warmup iterations 0..=65536",
        ));
    }
    report.warmup = Phase::new(
        "warmup",
        expert_sequence(report.args.warmup_iterations, NAMESPACE).map_err(Failure::accounting)?,
    );
    report.measured = Phase::new(
        "measured",
        expert_sequence(report.args.iterations, NAMESPACE).map_err(Failure::accounting)?,
    );
    // Read/hash/parse one identical config snapshot; never construct an Engine,
    // RealModel, residency manager, dense tensor loader or production backend.
    let bytes =
        std::fs::read(&report.args.config).map_err(|e| Failure::runtime("config-failed", e))?;
    report.config_sha256 = Some(sha(&bytes));
    let text = std::str::from_utf8(&bytes).map_err(|e| Failure::runtime("config-failed", e))?;
    let config: Config = toml::from_str(text).map_err(|e| Failure::runtime("config-failed", e))?;
    config.validate().map_err(Failure::authority)?;
    report.authority.direct_io_requested = !config.storage.no_direct;
    report.authority.packed_storage =
        Some(config.storage.packed_blob.is_some() || config.storage.packed_manifest.is_some());
    report.authority.source_data_dir = Some(config.model.data_dir.clone());
    validate_geometry(&config)?;
    report.authority.exact_geometry = true;
    if !report.authority.linux
        || report.args.expected_adapter_name != "NVIDIA L4"
        || !report.authority.direct_io_requested
        || report.authority.packed_storage != Some(false)
    {
        return Err(Failure::authority(
            "requires Linux, exact NVIDIA L4, direct I/O enabled, and non-packed per-expert layout",
        ));
    }
    let storage = NvmeStorage::new(StorageConfig {
        base_path: config.model.data_dir,
        expert_size: FULL,
        block_align: ALIGN,
        use_direct_io: true,
        num_experts_per_layer: Some(128),
    })
    .map_err(|e| Failure::runtime("source-failed", e))?;
    if storage.is_packed() {
        return Err(Failure::authority("packed storage is forbidden"));
    }
    let gpu = Gpu::new(&mut report.authority).await?;
    let pool = BufferPool::new(1, FULL, ALIGN);
    let mut buf = pool
        .try_acquire()
        .ok_or_else(|| Failure::runtime("runtime-failed", "CONTROL pool allocation unavailable"))?;
    run_phase(&mut report.warmup, &gpu, &storage, &mut buf).await?;
    run_phase(&mut report.measured, &gpu, &storage, &mut buf).await?;
    gpu.check()?;
    report.classify();
    Ok(())
}

pub(crate) async fn run_command(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Exclusive creation prevents accidentally replacing an immutable experiment.
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.report_out)?;
    let mut report = Report::new(args);
    let execution = std::panic::AssertUnwindSafe(execute(&mut report))
        .catch_unwind()
        .await;
    match execution {
        Ok(Ok(())) => {}
        Ok(Err(failure)) => report.fail(failure),
        Err(payload) => {
            let detail = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "non-string panic".into());
            report.fail(Failure::runtime(
                "runtime-failed",
                format!("diagnostic panic: {detail}"),
            ));
        }
    }
    serde_json::to_writer_pretty(&mut output, &report)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    if report.complete {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "{}: {}",
            report.classification,
            report.failure.as_deref().unwrap_or("diagnostic incomplete")
        ))
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn args() -> Args {
        Args {
            config: "unused.toml".into(),
            expected_adapter_name: "NVIDIA L4".into(),
            warmup_iterations: 0,
            iterations: 2,
            report_out: "unused.json".into(),
        }
    }
    fn config() -> Config {
        toml::from_str(
            r#"
[server]
[model]
data_dir = "/nonexistent/source-to-upload-test"
num_experts = 128
top_k = 8
d_model = 2048
d_ff = 768
expert_size = 2658304
num_layers = 48
dtype = "q4_0"
[storage]
cache_slots = 48
block_align = 4096
no_direct = false
"#,
        )
        .unwrap()
    }
    fn source(tag: u8) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(FULL);
        TensorHeader::for_swiglu_expert(WeightDtype::Q4_0, 2048, 768)
            .write_padded(ALIGN, &mut bytes);
        bytes.resize(FULL, tag);
        bytes
    }
    fn good_arm(n: u64, treatment: bool) -> Arm {
        let mut a = Arm {
            ops_attempted: n,
            source_read_attempts: n,
            source_read_ops: n,
            full_source_bytes: n * FULL as u64,
            payload_ops: n,
            payload_bytes: n * PAYLOAD as u64,
            cpu_payload_copy_bytes: if treatment { 0 } else { n * PAYLOAD as u64 },
            upload_ops: n,
            gpu_copied_bytes: n * PAYLOAD as u64,
            explicit_copy_buffer_bytes: if treatment { n * PAYLOAD as u64 } else { 0 },
            epoch_bytes: n * 4,
            gpu_completed_ops: n,
            verified_ops: n,
            verification_readback_bytes: n * SLOT as u64,
            verification_destination_reset_ops: n,
            verification_destination_reset_bytes: n * SLOT as u64,
            ..Arm::default()
        };
        a.fd_evidence = FdEvidence {
            checks: n,
            direct_observed: n,
            full_file_length_observed: n,
            ..FdEvidence::default()
        };
        a.pointers.observations = n;
        a.pointers.aligned = n;
        a.times.source_direct_read_ns = n * FULL as u64;
        a.times.transfer_cycle_ns = n * FULL as u64;
        if treatment {
            a.maps_completed = n;
            a.map_attempts = n;
            a.unmaps = n;
            a.pointers.gpu_offset_checks = n;
        }
        a
    }
    fn good_phase(name: &'static str, n: usize) -> Phase {
        let mut p = Phase::new(name, expert_sequence(n, NAMESPACE).unwrap());
        p.pairs_attempted = n as u64;
        p.pairs_completed = n as u64;
        p.attempted_expert_id_sequence_sha256 = p.expert_id_sequence_sha256.clone();
        p.control = good_arm(n as u64, false);
        p.treatment = good_arm(n as u64, true);
        p.control_witnesses = Witnesses {
            full_source_sha256: sha(b"source"),
            bare_payload_sha256: sha(b"payload"),
            gpu_destination_payload_sha256: sha(b"payload"),
        };
        p.treatment_witnesses = Witnesses {
            full_source_sha256: sha(b"source"),
            bare_payload_sha256: sha(b"payload"),
            gpu_destination_payload_sha256: sha(b"payload"),
        };
        p
    }
    fn good_report() -> Report {
        let mut r = Report::new(args());
        r.authority.linux = true;
        r.authority.adapter_authoritative = true;
        r.authority.direct_io_requested = true;
        r.authority.packed_storage = Some(false);
        r.authority.exact_geometry = true;
        r.warmup = good_phase("warmup", 0);
        r.measured = good_phase("measured", 2);
        r
    }
    fn rejected_arm(n: u64, errno: i32) -> Arm {
        let mut a = good_arm(n, true);
        a.source_read_ops = 0;
        a.full_source_bytes = 0;
        a.payload_ops = 0;
        a.payload_bytes = 0;
        a.upload_ops = 0;
        a.gpu_copied_bytes = 0;
        a.explicit_copy_buffer_bytes = 0;
        a.epoch_bytes = 0;
        a.gpu_completed_ops = 0;
        a.verified_ops = 0;
        a.verification_readback_bytes = 0;
        a.source_failures = n;
        a.mapped_direct_io_rejections = n;
        a.rejection_errno_counts.insert(errno, n);
        a
    }

    #[test]
    fn source_to_upload_alignment_smallest_offset_for_every_page_residue() {
        for residue in 0..ALIGN {
            let base = ALIGN * 16 + residue;
            let offset = aligned_subrange(base, UPLOAD, FULL, ALIGN).unwrap();
            assert_eq!((base + offset) % ALIGN, 0);
            assert!(offset < ALIGN && offset + FULL <= UPLOAD);
            if offset > 0 {
                assert_ne!((base + offset - 1) % ALIGN, 0);
            }
            assert_eq!(
                copy_offsets(offset, PREFIX, PAYLOAD, UPLOAD).is_ok(),
                residue % 4 == 0
            );
        }
    }
    #[test]
    fn source_to_upload_alignment_recomputed_after_remap() {
        assert_eq!(aligned_subrange(0x1000, UPLOAD, FULL, ALIGN).unwrap(), 0);
        assert_eq!(aligned_subrange(0x2008, UPLOAD, FULL, ALIGN).unwrap(), 4088);
        assert_eq!(copy_offsets(4088, PREFIX, PAYLOAD, UPLOAD).unwrap(), 8184);
    }
    #[test]
    fn source_to_upload_alignment_rejects_invalid_impossible_and_overflow() {
        for (base, capacity, len, align) in [
            (0, UPLOAD, FULL, ALIGN),
            (4096, UPLOAD, FULL, 0),
            (4096, UPLOAD, FULL, 3),
            (4096, UPLOAD, 0, ALIGN),
            (4096, UPLOAD, FULL - 1, ALIGN),
            (4097, FULL, FULL, ALIGN),
            (usize::MAX, UPLOAD, FULL, ALIGN),
        ] {
            assert!(aligned_subrange(base, capacity, len, align).is_err());
        }
        assert!(copy_offsets(1, PREFIX, PAYLOAD, UPLOAD).is_err());
        assert!(copy_offsets(0, PREFIX, PAYLOAD - 1, UPLOAD).is_err());
        assert!(copy_offsets(usize::MAX, PREFIX, PAYLOAD, UPLOAD).is_err());
        assert!(copy_offsets(ALIGN, PREFIX, PAYLOAD, FULL).is_err());
    }
    #[test]
    fn source_to_upload_exact_authoritative_geometry() {
        let mut c = config();
        c.validate().unwrap();
        validate_geometry(&c).unwrap();
        assert_eq!(3 * 2048 * 768 / 32 * 18, PAYLOAD);
        assert_eq!(FULL % ALIGN, 0);
        // Bare payload is also block aligned; only FULL satisfies the source contract.
        assert_eq!(PAYLOAD % ALIGN, 0);
        assert_eq!(SLOT, EPOCH_OFFSET + PAYLOAD);
        for size in [PAYLOAD, FULL - ALIGN, FULL + ALIGN] {
            c.model.expert_size = size;
            assert!(validate_geometry(&c).is_err());
        }
        c = config();
        c.model.d_ff += 32;
        assert!(validate_geometry(&c).is_err());
        c = config();
        c.model.num_layers -= 1;
        assert!(validate_geometry(&c).is_err());
    }
    #[test]
    fn source_to_upload_header_offsets_and_validation() {
        let bytes = source(7);
        let (offset, payload) = payload_range(&bytes).unwrap();
        assert_eq!(offset, PREFIX);
        assert_eq!(payload.len(), PAYLOAD);
        assert_eq!(payload.as_ptr() as usize - bytes.as_ptr() as usize, PREFIX);
        for index in [0, 4, 6, 7, 8, 12, 44] {
            let mut bad = bytes.clone();
            bad[index] ^= 0xff;
            assert!(payload_range(&bad).is_err(), "header byte {index}");
        }
        assert!(payload_range(&bytes[..FULL - 1]).is_err());
    }
    #[test]
    fn source_to_upload_sequence_spans_namespace_and_is_deterministic() {
        let ids = expert_sequence(128, NAMESPACE).unwrap();
        assert_eq!(ids.first(), Some(&0));
        assert_eq!(ids.last(), Some(&6143));
        assert_eq!(ids, expert_sequence(128, NAMESPACE).unwrap());
        assert!(ids.windows(2).all(|p| p[0] < p[1]));
        let all = expert_sequence(6144, NAMESPACE).unwrap();
        assert_eq!(all, (0..NAMESPACE).collect::<Vec<_>>());
        let repeated = expert_sequence(6146, NAMESPACE).unwrap();
        assert_eq!(&repeated[6144..], &[0, 1]);
        assert_eq!(expert_sequence(1, NAMESPACE).unwrap(), vec![3072]);
        assert!(expert_sequence(2, 1).is_err());
        assert!(expert_sequence(MAX_ITERATIONS + 1, NAMESPACE).is_err());
    }
    #[test]
    fn source_to_upload_sequence_hash_frozen_little_endian() {
        let ids = expert_sequence(4, NAMESPACE).unwrap();
        assert_eq!(ids, vec![0, 2047, 4095, 6143]);
        // Independent known-byte encoding, including IDs exceeding one byte.
        assert_eq!(
            sequence_sha(&ids),
            sha(&[0, 0, 0, 0, 255, 7, 0, 0, 255, 15, 0, 0, 255, 23, 0, 0])
        );
        let mut reversed = ids.clone();
        reversed.reverse();
        assert_ne!(sequence_sha(&ids), sequence_sha(&reversed));
        assert_ne!(sequence_sha(&[1, 1]), sequence_sha(&[1]));
    }
    #[test]
    fn source_to_upload_witnesses_hash_exact_ordered_bytes_without_payload_copies() {
        let a = source(7);
        let b = source(23);
        let mut streams = Streams::default();
        streams.source(&a).unwrap();
        streams.source(&b).unwrap();
        streams.gpu.update(&a[PREFIX..]);
        streams.gpu.update(&b[PREFIX..]);
        let w = streams.snapshot();
        assert_eq!(
            w.full_source_sha256,
            sha(&[a.as_slice(), b.as_slice()].concat())
        );
        assert_eq!(
            w.bare_payload_sha256,
            sha(&[&a[PREFIX..], &b[PREFIX..]].concat())
        );
        assert_eq!(w.bare_payload_sha256, w.gpu_destination_payload_sha256);
        let mut reversed = Streams::default();
        reversed.source(&b).unwrap();
        reversed.source(&a).unwrap();
        assert_ne!(w.full_source_sha256, reversed.snapshot().full_source_sha256);
    }
    #[test]
    fn source_to_upload_pair_mismatch_records_first_context() {
        let mut p = good_phase("measured", 2);
        let c = Outcome::Verified(Hashes {
            source: "source-a".into(),
            payload: "p".into(),
            gpu: "p".into(),
            epoch: true,
        });
        let t = Outcome::Verified(Hashes {
            source: "source-b".into(),
            payload: "p".into(),
            gpu: "p".into(),
            epoch: true,
        });
        p.compare(1, 6143, &c, &t).unwrap();
        assert_eq!(p.mismatch_count, 1);
        let first = p.first_mismatch.as_ref().unwrap();
        assert_eq!(
            (first.pair, first.expert_id, first.kind),
            (1, 6143, "full-source")
        );
        p.compare(2, 0, &c, &t).unwrap();
        assert_eq!(p.mismatch_count, 2);
        assert_eq!(p.first_mismatch.as_ref().unwrap().expert_id, 6143);
    }
    #[test]
    fn source_to_upload_pass_does_not_require_total_cycle_superiority() {
        let mut r = good_report();
        r.measured.treatment.times.transfer_cycle_ns *= 10;
        r.classify();
        assert!(r.complete && r.feasibility_pass);
        assert_eq!(r.classification, "copy-elision-feasible");
        assert!(
            r.measured.treatment.transfer_cycle_payload_gbps
                < r.measured.control.transfer_cycle_payload_gbps
        );
    }
    #[test]
    fn source_to_upload_source_only_ninety_percent_boundary() {
        let mut r = good_report();
        r.measured.control.times.source_direct_read_ns = 900;
        r.measured.treatment.times.source_direct_read_ns = 1000;
        r.measured.treatment.times.verification_readback_ns = u64::MAX / 2;
        r.classify();
        assert!(r.feasibility_pass);
        let mut r = good_report();
        r.measured.control.times.source_direct_read_ns = 899;
        r.measured.treatment.times.source_direct_read_ns = 1000;
        r.classify();
        assert_eq!(r.classification, "mapped-source-too-slow");
        assert!(r.complete && !r.feasibility_pass);
    }
    #[test]
    fn source_to_upload_gate_fails_closed_on_authority_or_evidence_mutation() {
        let mutations: Vec<fn(&mut Report)> = vec![
            |r| r.authority.linux = false,
            |r| r.authority.adapter_authoritative = false,
            |r| r.authority.expected_adapter_name = "other".into(),
            |r| r.authority.direct_io_requested = false,
            |r| r.authority.packed_storage = Some(true),
            |r| r.authority.exact_geometry = false,
            |r| r.measured.control.fd_evidence.direct_observed -= 1,
            |r| r.measured.treatment.fd_evidence.full_file_length_observed -= 1,
            |r| r.measured.treatment.pointers.aligned -= 1,
            |r| r.measured.treatment.fallback_reads = 1,
            |r| r.measured.control.cpu_payload_copy_bytes -= 1,
            |r| r.measured.treatment.cpu_payload_copy_bytes = 1,
            |r| r.measured.treatment.explicit_copy_buffer_bytes -= 4,
            |r| r.measured.treatment.exact_read_length_failures = 1,
            |r| r.measured.treatment.pointers.gpu_offset_failures = 1,
            |r| r.measured.treatment.source_failures = 1,
            |r| r.measured.treatment.gpu_failures = 1,
            |r| r.measured.treatment.map_failures = 1,
            |r| r.measured.treatment.accounting_failures = 1,
            |r| r.measured.treatment.unmaps -= 1,
            |r| r.measured.treatment.verification_destination_reset_ops -= 1,
            |r| r.measured.control.verified_ops -= 1,
            |r| r.measured.attempted_expert_id_sequence_sha256 = "wrong".into(),
            |r| r.measured.treatment_witnesses.bare_payload_sha256 = "wrong".into(),
            |r| r.measured.control.times.source_direct_read_ns = 0,
            |r| r.runtime_failures = 1,
        ];
        for (i, mutate) in mutations.into_iter().enumerate() {
            let mut r = good_report();
            mutate(&mut r);
            r.classify();
            assert!(!r.feasibility_pass, "mutation {i}");
        }
    }
    #[test]
    fn source_to_upload_treatment_cpu_copy_accounting_is_exactly_zero() {
        let mut arm = good_arm(2, true);
        assert!(arm.reconcile(true));
        arm.cpu_payload_copy_bytes = 4;
        assert!(!arm.reconcile(true));
        let mut control = good_arm(2, false);
        control.cpu_payload_copy_bytes += 1;
        assert!(!control.reconcile(false));
        let mut overflow = good_arm(2, false);
        overflow.source_read_ops = u64::MAX;
        assert!(!overflow.reconcile(false));
    }
    #[test]
    fn source_to_upload_negative_direct_io_rejection_is_complete() {
        for errno in [libc::EINVAL, libc::EFAULT] {
            let mut r = good_report();
            r.measured.treatment = rejected_arm(2, errno);
            r.classify();
            assert_eq!(r.classification, "mapped-upload-direct-io-rejected");
            assert!(r.complete && !r.feasibility_pass);
        }
        assert!(!mapped_rejection(Some(libc::EIO)));
        assert!(!mapped_rejection(None));
    }
    #[test]
    fn source_to_upload_negative_rejection_requires_working_control_and_consistency() {
        let mut r = good_report();
        r.measured.treatment = rejected_arm(2, libc::EINVAL);
        r.measured.control.source_failures = 1;
        r.classify();
        assert!(!r.complete && !r.feasibility_pass);
        let mut r = good_report();
        r.measured.treatment = rejected_arm(2, libc::EIO);
        r.classify();
        assert!(!r.complete && !r.feasibility_pass);
        let mut r = good_report();
        r.measured.treatment = rejected_arm(2, libc::EINVAL);
        r.measured.treatment.mapped_direct_io_rejections = 1;
        r.classify();
        assert!(!r.complete);
        let mut r = good_report();
        r.warmup = good_phase("warmup", 1);
        r.measured.treatment = rejected_arm(2, libc::EINVAL);
        r.classify();
        assert!(!r.complete);
    }
    #[test]
    fn source_to_upload_alignment_unavailable_is_scientific_negative() {
        let mut r = good_report();
        let mut t = rejected_arm(2, libc::EINVAL);
        t.source_read_attempts = 0;
        t.pointers.aligned = 0;
        t.pointers.observations = 0;
        t.source_failures = 0;
        t.mapped_direct_io_rejections = 0;
        t.rejection_errno_counts.clear();
        t.alignment_failures = 2;
        r.measured.treatment = t;
        r.classify();
        assert_eq!(r.classification, "alignment-contract-unavailable");
        assert!(r.complete && !r.feasibility_pass);
    }
    #[test]
    fn source_to_upload_parity_classifications() {
        for (kind, expected) in [
            ("full-source", "source-parity-failed"),
            ("bare-payload", "source-parity-failed"),
            ("treatment-gpu-payload", "gpu-copy-parity-failed"),
        ] {
            let mut r = good_report();
            r.measured.mismatch(0, 0, kind, "a", "b").unwrap();
            r.classify();
            assert_eq!(r.classification, expected);
            assert!(r.complete && !r.feasibility_pass);
        }
    }
    #[test]
    fn source_to_upload_read_errors_preserve_errno_and_reject_short_reads() {
        let mut a = Arm::default();
        let mut context = None;
        assert!(!read_result(
            Err(io::Error::from_raw_os_error(libc::EINVAL)),
            true,
            &mut a,
            17,
            &mut context
        )
        .unwrap());
        assert_eq!(a.source_failures, 1);
        assert_eq!(a.rejection_errno_counts[&libc::EINVAL], 1);
        assert!(context.unwrap().contains("expert 17"));
        assert!(read_result(
            Err(io::Error::from_raw_os_error(libc::EINVAL)),
            false,
            &mut a,
            17,
            &mut None
        )
        .is_err());
        assert!(read_result(Ok(FULL - 1), true, &mut a, 17, &mut None).is_err());
        assert_eq!(a.exact_read_length_failures, 1);
    }
    #[test]
    fn source_to_upload_cli_parses_explicit_and_defaults_without_startup_model_loading() {
        let cli = crate::Cli::try_parse_from([
            "mer",
            "diagnose-gpu-native-source-to-upload-copy-elision",
            "--config",
            "frozen.toml",
            "--report-out",
            "new.json",
        ])
        .unwrap();
        assert!(crate::startup_config_path(&cli.cmd).is_none());
        assert!(matches!(
            cli.cmd,
            crate::Cmd::DiagnoseGpuNativeSourceToUploadCopyElision {
                iterations: 128,
                warmup_iterations: 3,
                ..
            }
        ));
        let cli = crate::Cli::try_parse_from([
            "mer",
            "diagnose-gpu-native-source-to-upload-copy-elision",
            "--config",
            "frozen.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--warmup-iterations",
            "0",
            "--iterations",
            "16",
            "--report-out",
            "new.json",
        ])
        .unwrap();
        assert!(matches!(
            cli.cmd,
            crate::Cmd::DiagnoseGpuNativeSourceToUploadCopyElision {
                iterations: 16,
                warmup_iterations: 0,
                ..
            }
        ));
        assert!(crate::Cli::try_parse_from([
            "mer",
            "diagnose-gpu-native-source-to-upload-copy-elision"
        ])
        .is_err());
        assert!(crate::Cli::try_parse_from([
            "mer",
            "diagnose-gpu-native-source-to-upload-copy-elision",
            "--config",
            "frozen.toml",
            "--report-out",
            "new.json",
            "--no-direct"
        ])
        .is_err());
    }
    #[test]
    fn source_to_upload_report_completion_semantics() {
        let mut r = good_report();
        r.fail(Failure::authority("wrong adapter"));
        assert!(r.complete && !r.feasibility_pass);
        assert_eq!(r.runtime_failures, 0);
        r.fail(Failure::runtime("gpu-failed", "unexpected failure"));
        assert!(!r.complete && !r.feasibility_pass);
        assert_eq!(r.runtime_failures, 1);
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["schema"], SCHEMA);
        assert_eq!(json["complete"], false);
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn source_to_upload_runner_writes_incomplete_failure_and_protects_report() {
        let dir = std::env::temp_dir().join(format!(
            "mer-source-to-upload-runner-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut a = args();
        a.config = dir.join("missing.toml");
        a.report_out = dir.join("failure.json");
        assert!(run_command(a.clone()).await.is_err());
        let bytes = std::fs::read(&a.report_out).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["complete"], false);
        assert_eq!(json["classification"], "config-failed");
        assert!(run_command(a.clone()).await.is_err());
        assert_eq!(std::fs::read(&a.report_out).unwrap(), bytes);
        a.config = dir.join("valid.toml");
        a.report_out = dir.join("authority.json");
        a.expected_adapter_name = "forbidden-test-adapter".into();
        std::fs::write(&a.config, toml::to_string(&config()).unwrap()).unwrap();
        run_command(a.clone()).await.unwrap();
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&a.report_out).unwrap()).unwrap();
        assert_eq!(json["complete"], true);
        assert_eq!(json["feasibility_pass"], false);
        assert_eq!(json["classification"], "authority-failed");
        assert!(json["authority"]["adapter_name"].is_null());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
