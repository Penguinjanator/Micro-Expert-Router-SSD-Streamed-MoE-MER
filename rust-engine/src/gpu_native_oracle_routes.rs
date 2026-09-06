//! Qualification-only capture of the ordered routes already selected by the
//! ordinary production GPU-native token loop, plus pure offline residency
//! lower-bound analysis.
//!
//! The command in this module calls only `GpuNativeTokenLoop::step_token`.
//! Route IDs are observed at `Engine::record_gpu_native_actual_routes`, after
//! the compact boundary report has already been read for normal recovery and
//! completion accounting. No diagnostic GPU copy or readback is added.

#[cfg(test)]
use crate::backend::gpu_native::GpuNativeQ4ExpertGeometry;
use crate::backend::GpuDeviceIdentity;
use crate::engine::{GpuNativeActualRouteObserver, RoutedExpertExecutionSnapshot};
use crate::gpu_native_real_benchmark::{
    BenchmarkFailure, BenchmarkProvenance, ProductionConfiguration, RequestEvidence,
    RuntimeContractEvidence, RuntimeContractInput,
};
use crate::gpu_native_residency::GpuNativeModelExpertVramPlan;
use crate::gpu_native_token_loop::{
    GpuNativeModelGeometry, GpuNativeRecoverySnapshot, GpuNativeTokenLoopSnapshot,
};
use crate::greedy_parity::{BackgroundShutdownEvidence, ModelIdentityEvidence, ModelLoadEvidence};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

pub(crate) const SCHEMA: &str = "mer.gpu-native-oracle-route-trace.v1";
pub(crate) const MODE: &str = "gpu-native-oracle-route-trace";
pub(crate) const BASE_GIT_SHA: &str = "7ab35f10d2798faf3d997e0d90f51f3e6519529e";
const CANONICAL_SERIALIZATION: &str = "domain mer.gpu-native-oracle-route-sequence.v1 NUL; record_count u64-le; repeated position u64-le, layer_index u32-le, selected_count u32-le, ordered local expert ids u32-le";
const LEGACY_SERIALIZATION: &str = "historical QualificationOrderedHasher sets: 0x53, set_len u64-le, ordered global expert ids u32-le; one set per position/layer";
const MIB: u64 = 1024 * 1024;
const EXPECTED_ADAPTER_NAME: &str = "NVIDIA L4";
const EXPECTED_EXPERT_FILE_BYTES: usize = 2_658_304;
const EXPECTED_ROUTED_EXPERT_FOOTPRINT_BYTES: u64 = 16_332_619_776;
const REQUESTED_BUDGETS_MIB: [u64; 7] = [2_048, 4_096, 8_192, 10_240, 12_288, 14_336, 15_576];

#[derive(Clone, Debug)]
pub(crate) struct CommandArgs {
    pub(crate) config: PathBuf,
    pub(crate) request_json: PathBuf,
    pub(crate) report_out: PathBuf,
    pub(crate) progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OracleGeometry {
    pub(crate) layers: usize,
    pub(crate) experts: usize,
    pub(crate) top_k: usize,
    pub(crate) d_model: usize,
    pub(crate) d_ff: usize,
}

impl From<GpuNativeModelGeometry> for OracleGeometry {
    fn from(value: GpuNativeModelGeometry) -> Self {
        Self {
            layers: value.num_layers,
            experts: value.num_experts,
            top_k: value.top_k,
            d_model: value.d_model,
            d_ff: value.d_ff,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OracleRouteRecord {
    pub(crate) position: usize,
    pub(crate) layer_index: usize,
    pub(crate) ordered_selected_expert_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OracleRouteTrace {
    pub(crate) schema: String,
    pub(crate) geometry: OracleGeometry,
    pub(crate) total_positions: usize,
    pub(crate) total_layer_records: usize,
    pub(crate) total_selected_expert_ids: usize,
    pub(crate) canonical_serialization: String,
    pub(crate) ordered_route_sequence_sha256: String,
    pub(crate) legacy_serialization: String,
    pub(crate) legacy_selected_route_ids_sha256: String,
    pub(crate) records: Vec<OracleRouteRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OracleTraceError {
    pub(crate) code: &'static str,
    pub(crate) detail: String,
}

impl OracleTraceError {
    fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for OracleTraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for OracleTraceError {}

fn validate_route_records(
    geometry: OracleGeometry,
    records: &[OracleRouteRecord],
) -> Result<(usize, usize), OracleTraceError> {
    if geometry.layers == 0
        || geometry.experts == 0
        || geometry.top_k == 0
        || geometry.top_k > geometry.experts
    {
        return Err(OracleTraceError::new(
            "invalid-geometry",
            format!("invalid oracle route geometry {geometry:?}"),
        ));
    }
    if records.is_empty() || !records.len().is_multiple_of(geometry.layers) {
        return Err(OracleTraceError::new(
            "wrong-layer-count",
            format!(
                "{} route records cannot form nonempty positions of exactly {} layers",
                records.len(),
                geometry.layers
            ),
        ));
    }

    let total_positions = records.len() / geometry.layers;
    let mut total_selected = 0usize;
    for position in 0..total_positions {
        for layer_index in 0..geometry.layers {
            let record_index = position
                .checked_mul(geometry.layers)
                .and_then(|base| base.checked_add(layer_index))
                .ok_or_else(|| {
                    OracleTraceError::new("route-count-overflow", "route index overflow")
                })?;
            let record = &records[record_index];
            if record.position != position || record.layer_index != layer_index {
                return Err(OracleTraceError::new(
                    "route-order",
                    format!(
                        "record {record_index} expected position={position} layer={layer_index}, observed position={} layer={}",
                        record.position, record.layer_index
                    ),
                ));
            }
            if record.ordered_selected_expert_ids.len() != geometry.top_k {
                return Err(OracleTraceError::new(
                    "wrong-top-k-count",
                    format!(
                        "position {position} layer {layer_index} has {} selected IDs, expected {}",
                        record.ordered_selected_expert_ids.len(),
                        geometry.top_k
                    ),
                ));
            }
            let mut unique = HashSet::with_capacity(geometry.top_k);
            for &expert_id in &record.ordered_selected_expert_ids {
                if expert_id as usize >= geometry.experts {
                    return Err(OracleTraceError::new(
                        "invalid-expert-id",
                        format!(
                            "position {position} layer {layer_index} selected expert {expert_id}, maximum is {}",
                            geometry.experts - 1
                        ),
                    ));
                }
                if !unique.insert(expert_id) {
                    return Err(OracleTraceError::new(
                        "duplicate-top-k-id",
                        format!(
                            "position {position} layer {layer_index} repeats expert {expert_id}"
                        ),
                    ));
                }
            }
            total_selected = total_selected
                .checked_add(record.ordered_selected_expert_ids.len())
                .ok_or_else(|| {
                    OracleTraceError::new("route-count-overflow", "selected expert count overflow")
                })?;
        }
    }
    Ok((total_positions, total_selected))
}

fn canonical_route_bytes(records: &[OracleRouteRecord]) -> Result<Vec<u8>, OracleTraceError> {
    let mut bytes = b"mer.gpu-native-oracle-route-sequence.v1\0".to_vec();
    let record_count = u64::try_from(records.len()).map_err(|_| {
        OracleTraceError::new("route-count-overflow", "record count does not fit u64")
    })?;
    bytes.extend_from_slice(&record_count.to_le_bytes());
    for record in records {
        let position = u64::try_from(record.position).map_err(|_| {
            OracleTraceError::new("route-count-overflow", "position does not fit u64")
        })?;
        let layer = u32::try_from(record.layer_index).map_err(|_| {
            OracleTraceError::new("route-count-overflow", "layer index does not fit u32")
        })?;
        let selected_count =
            u32::try_from(record.ordered_selected_expert_ids.len()).map_err(|_| {
                OracleTraceError::new("route-count-overflow", "top-k count does not fit u32")
            })?;
        bytes.extend_from_slice(&position.to_le_bytes());
        bytes.extend_from_slice(&layer.to_le_bytes());
        bytes.extend_from_slice(&selected_count.to_le_bytes());
        for &expert_id in &record.ordered_selected_expert_ids {
            bytes.extend_from_slice(&expert_id.to_le_bytes());
        }
    }
    Ok(bytes)
}

fn ordered_route_sequence_sha256(
    records: &[OracleRouteRecord],
) -> Result<String, OracleTraceError> {
    Ok(format!(
        "{:x}",
        Sha256::digest(canonical_route_bytes(records)?)
    ))
}

fn legacy_route_sequence_sha256(
    geometry: OracleGeometry,
    records: &[OracleRouteRecord],
) -> Result<String, OracleTraceError> {
    let mut global_sets = Vec::with_capacity(records.len());
    for record in records {
        let layer_base = record
            .layer_index
            .checked_mul(geometry.experts)
            .ok_or_else(|| {
                OracleTraceError::new("route-count-overflow", "global layer offset overflow")
            })?;
        let ids = record
            .ordered_selected_expert_ids
            .iter()
            .map(|&local_id| {
                layer_base
                    .checked_add(local_id as usize)
                    .and_then(|global_id| u32::try_from(global_id).ok())
                    .ok_or_else(|| {
                        OracleTraceError::new("route-count-overflow", "global expert ID overflow")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        global_sets.push(ids);
    }
    Ok(crate::engine::qualification_ordered_sets_sha256(
        &global_sets,
    ))
}

impl OracleRouteTrace {
    pub(crate) fn try_new(
        geometry: OracleGeometry,
        records: Vec<OracleRouteRecord>,
    ) -> Result<Self, OracleTraceError> {
        let (total_positions, total_selected_expert_ids) =
            validate_route_records(geometry, &records)?;
        let total_layer_records = records.len();
        let ordered_route_sequence_sha256 = ordered_route_sequence_sha256(&records)?;
        let legacy_selected_route_ids_sha256 = legacy_route_sequence_sha256(geometry, &records)?;
        Ok(Self {
            schema: SCHEMA.to_string(),
            geometry,
            total_positions,
            total_layer_records,
            total_selected_expert_ids,
            canonical_serialization: CANONICAL_SERIALIZATION.to_string(),
            ordered_route_sequence_sha256,
            legacy_serialization: LEGACY_SERIALIZATION.to_string(),
            legacy_selected_route_ids_sha256,
            records,
        })
    }

    pub(crate) fn validate(&self) -> Result<(), OracleTraceError> {
        if self.schema != SCHEMA {
            return Err(OracleTraceError::new(
                "wrong-schema",
                format!("expected {SCHEMA}, observed {}", self.schema),
            ));
        }
        if self.canonical_serialization != CANONICAL_SERIALIZATION
            || self.legacy_serialization != LEGACY_SERIALIZATION
        {
            return Err(OracleTraceError::new(
                "serialization-contract-drift",
                "route hash serialization description does not match schema v1",
            ));
        }
        let (positions, selected) = validate_route_records(self.geometry, &self.records)?;
        if self.total_positions != positions
            || self.total_layer_records != self.records.len()
            || self.total_selected_expert_ids != selected
        {
            return Err(OracleTraceError::new(
                "trace-total-mismatch",
                "stored trace totals disagree with validated records",
            ));
        }
        let canonical = ordered_route_sequence_sha256(&self.records)?;
        if self.ordered_route_sequence_sha256 != canonical {
            return Err(OracleTraceError::new(
                "canonical-route-hash-mismatch",
                format!(
                    "stored {} differs from recomputed {canonical}",
                    self.ordered_route_sequence_sha256
                ),
            ));
        }
        let legacy = legacy_route_sequence_sha256(self.geometry, &self.records)?;
        if self.legacy_selected_route_ids_sha256 != legacy {
            return Err(OracleTraceError::new(
                "legacy-route-hash-mismatch",
                format!(
                    "stored {} differs from recomputed {legacy}",
                    self.legacy_selected_route_ids_sha256
                ),
            ));
        }
        Ok(())
    }

    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, OracleTraceError> {
        let trace: Self = serde_json::from_slice(bytes)
            .map_err(|error| OracleTraceError::new("trace-parse-failed", error.to_string()))?;
        trace.validate()?;
        Ok(trace)
    }
}

#[derive(Default)]
struct RouteCollector {
    records: parking_lot::Mutex<Vec<OracleRouteRecord>>,
}

impl GpuNativeActualRouteObserver for RouteCollector {
    fn record_position(&self, position: usize, selected_ids_by_layer: &[Vec<u32>]) {
        let mut records = self.records.lock();
        records.extend(
            selected_ids_by_layer
                .iter()
                .enumerate()
                .map(|(layer_index, ids)| OracleRouteRecord {
                    position,
                    layer_index,
                    ordered_selected_expert_ids: ids.clone(),
                }),
        );
    }
}

impl RouteCollector {
    fn take(&self) -> Vec<OracleRouteRecord> {
        std::mem::take(&mut *self.records.lock())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PolicyCounts {
    route_references: u64,
    cache_hits: u64,
    missing_installs: u64,
    miss_boundaries: u64,
    compulsory_cold_installs: u64,
}

fn validate_accesses(
    capacity: usize,
    experts: usize,
    accesses: &[Vec<u32>],
) -> Result<(), OracleTraceError> {
    if capacity == 0 {
        return Err(OracleTraceError::new(
            "zero-slot-capacity",
            "offline residency analysis requires at least one physical slot",
        ));
    }
    for (position, ids) in accesses.iter().enumerate() {
        if ids.len() > capacity {
            return Err(OracleTraceError::new(
                "route-set-exceeds-capacity",
                format!(
                    "position {position} needs {} simultaneous experts but capacity is {capacity}",
                    ids.len()
                ),
            ));
        }
        let mut unique = HashSet::with_capacity(ids.len());
        for &id in ids {
            if id as usize >= experts || !unique.insert(id) {
                return Err(OracleTraceError::new(
                    "invalid-analysis-route",
                    format!("position {position} contains invalid or duplicate expert {id}"),
                ));
            }
        }
    }
    Ok(())
}

/// Exact empty-start LRU simulation of production demand-set semantics: hits
/// are touched in ordered top-k order, the whole set is protected while enough
/// oldest unprotected residents are evicted, then misses install in route order.
fn simulate_production_lru(
    capacity: usize,
    experts: usize,
    accesses: &[Vec<u32>],
) -> Result<PolicyCounts, OracleTraceError> {
    validate_accesses(capacity, experts, accesses)?;
    let mut lru = VecDeque::<u32>::with_capacity(capacity);
    let mut ever_seen = HashSet::new();
    let mut counts = PolicyCounts::default();
    for ids in accesses {
        counts.route_references = counts
            .route_references
            .checked_add(ids.len() as u64)
            .ok_or_else(|| OracleTraceError::new("analysis-overflow", "route count overflow"))?;
        let protected = ids.iter().copied().collect::<HashSet<_>>();
        let mut misses = Vec::new();
        for &id in ids {
            if let Some(index) = lru.iter().position(|&resident| resident == id) {
                let resident = lru.remove(index).expect("located LRU resident");
                lru.push_back(resident);
                counts.cache_hits += 1;
            } else {
                misses.push(id);
                counts.missing_installs += 1;
                if ever_seen.insert(id) {
                    counts.compulsory_cold_installs += 1;
                }
            }
        }
        if !misses.is_empty() {
            counts.miss_boundaries += 1;
        }
        while lru.len().saturating_add(misses.len()) > capacity {
            let victim = lru
                .iter()
                .position(|id| !protected.contains(id))
                .ok_or_else(|| {
                    OracleTraceError::new(
                        "no-unprotected-lru-victim",
                        "validated simultaneous set left no legal LRU victim",
                    )
                })?;
            lru.remove(victim);
        }
        for id in misses {
            lru.push_back(id);
        }
    }
    Ok(counts)
}

/// Belady/MIN-style empty-start simulation for simultaneous top-k accesses.
/// Every current route is protected; among other residents, the expert whose
/// next route-set use is farthest away (or absent) is evicted.
fn simulate_belady_min(
    capacity: usize,
    experts: usize,
    accesses: &[Vec<u32>],
) -> Result<PolicyCounts, OracleTraceError> {
    validate_accesses(capacity, experts, accesses)?;
    let mut future = vec![VecDeque::<usize>::new(); experts];
    for (position, ids) in accesses.iter().enumerate() {
        for &id in ids {
            future[id as usize].push_back(position);
        }
    }
    let mut residents = HashSet::<u32>::with_capacity(capacity);
    let mut ever_seen = HashSet::new();
    let mut counts = PolicyCounts::default();
    for (position, ids) in accesses.iter().enumerate() {
        for &id in ids {
            let observed = future[id as usize].pop_front();
            if observed != Some(position) {
                return Err(OracleTraceError::new(
                    "future-index-corrupt",
                    "next-use index did not match the current position",
                ));
            }
        }
        counts.route_references = counts
            .route_references
            .checked_add(ids.len() as u64)
            .ok_or_else(|| OracleTraceError::new("analysis-overflow", "route count overflow"))?;
        let protected = ids.iter().copied().collect::<HashSet<_>>();
        let mut misses = Vec::new();
        for &id in ids {
            if residents.contains(&id) {
                counts.cache_hits += 1;
            } else {
                misses.push(id);
                counts.missing_installs += 1;
                if ever_seen.insert(id) {
                    counts.compulsory_cold_installs += 1;
                }
            }
        }
        if !misses.is_empty() {
            counts.miss_boundaries += 1;
        }
        while residents.len().saturating_add(misses.len()) > capacity {
            let victim = residents
                .iter()
                .copied()
                .filter(|id| !protected.contains(id))
                .max_by_key(|id| {
                    (
                        future[*id as usize].front().copied().unwrap_or(usize::MAX),
                        *id,
                    )
                })
                .ok_or_else(|| {
                    OracleTraceError::new(
                        "no-unprotected-min-victim",
                        "validated simultaneous set left no legal MIN victim",
                    )
                })?;
            residents.remove(&victim);
        }
        residents.extend(misses);
    }
    Ok(counts)
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LayerOracleAnalysis {
    pub(crate) layer_index: usize,
    pub(crate) physical_expert_slots: usize,
    pub(crate) route_references: u64,
    pub(crate) unique_experts_touched: usize,
    pub(crate) lru_cache_hits: u64,
    pub(crate) lru_missing_expert_installs: u64,
    pub(crate) lru_layer_miss_boundaries: u64,
    pub(crate) lru_bytes_to_move: u64,
    pub(crate) minimum_cache_hits: u64,
    pub(crate) minimum_missing_expert_installs: u64,
    pub(crate) minimum_layer_miss_boundaries: u64,
    pub(crate) minimum_bytes_to_move: u64,
    pub(crate) compulsory_cold_installs: u64,
    pub(crate) lru_capacity_replacement_installs: u64,
    pub(crate) minimum_capacity_replacement_installs: u64,
    pub(crate) lru_resident_route_fraction: f64,
    pub(crate) maximum_potentially_resident_route_fraction: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct AggregateOracleAnalysis {
    pub(crate) physical_expert_slots: usize,
    pub(crate) route_references: u64,
    pub(crate) unique_experts_touched: usize,
    pub(crate) lru_cache_hits: u64,
    pub(crate) lru_missing_expert_installs: u64,
    pub(crate) lru_layer_miss_boundaries: u64,
    pub(crate) lru_bytes_to_move: u64,
    pub(crate) minimum_cache_hits: u64,
    pub(crate) minimum_missing_expert_installs: u64,
    pub(crate) minimum_layer_miss_boundaries: u64,
    pub(crate) minimum_bytes_to_move: u64,
    pub(crate) compulsory_cold_installs: u64,
    pub(crate) lru_capacity_replacement_installs: u64,
    pub(crate) minimum_capacity_replacement_installs: u64,
    pub(crate) lru_resident_route_fraction: f64,
    pub(crate) maximum_potentially_resident_route_fraction: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BudgetOracleAnalysis {
    pub(crate) requested_expert_budget_mib: u64,
    pub(crate) requested_expert_budget_bytes: u64,
    pub(crate) minimum_executable_budget_bytes: u64,
    pub(crate) arena_allocation_bytes: u64,
    pub(crate) unused_budget_bytes: u64,
    pub(crate) physical_slot_stride_bytes: u64,
    pub(crate) physical_expert_slots: usize,
    pub(crate) per_layer_physical_expert_slots: Vec<usize>,
    pub(crate) per_layer: Vec<LayerOracleAnalysis>,
    pub(crate) aggregate: AggregateOracleAnalysis,
}

fn checked_transfer_bytes(installs: u64, bytes: u64) -> Result<u64, OracleTraceError> {
    installs
        .checked_mul(bytes)
        .ok_or_else(|| OracleTraceError::new("analysis-overflow", "transfer byte count overflow"))
}

fn resident_fraction(hits: u64, references: u64) -> f64 {
    if references == 0 {
        1.0
    } else {
        hits as f64 / references as f64
    }
}

fn analyze_trace_with_plan(
    trace: &OracleRouteTrace,
    requested_budget_mib: u64,
    plan: &GpuNativeModelExpertVramPlan,
) -> Result<BudgetOracleAnalysis, OracleTraceError> {
    trace.validate()?;
    let plan_geometry = plan.geometry();
    if plan.num_layers() != trace.geometry.layers
        || plan_geometry.num_experts() != trace.geometry.experts
        || plan_geometry.top_k() != trace.geometry.top_k
        || plan_geometry.d_model() != trace.geometry.d_model
        || plan_geometry.d_ff() != trace.geometry.d_ff
    {
        return Err(OracleTraceError::new(
            "slot-plan-geometry-mismatch",
            "production slot-plan geometry differs from the validated route trace",
        ));
    }
    let capacities = plan
        .layer_plans()
        .iter()
        .map(|layer| layer.slot_capacity())
        .collect::<Vec<_>>();
    let slot_stride_bytes = plan_geometry.slot_stride_bytes() as u64;
    let mut accesses = vec![Vec::<Vec<u32>>::new(); trace.geometry.layers];
    for record in &trace.records {
        accesses[record.layer_index].push(record.ordered_selected_expert_ids.clone());
    }

    let mut aggregate = AggregateOracleAnalysis::default();
    let mut per_layer = Vec::with_capacity(trace.geometry.layers);
    for layer_index in 0..trace.geometry.layers {
        let capacity = capacities[layer_index];
        let layer_accesses = &accesses[layer_index];
        let unique_experts_touched = layer_accesses
            .iter()
            .flatten()
            .copied()
            .collect::<HashSet<_>>()
            .len();
        let lru = simulate_production_lru(capacity, trace.geometry.experts, layer_accesses)?;
        let minimum = simulate_belady_min(capacity, trace.geometry.experts, layer_accesses)?;
        if lru.route_references != minimum.route_references
            || lru.compulsory_cold_installs != minimum.compulsory_cold_installs
        {
            return Err(OracleTraceError::new(
                "analysis-reconciliation",
                "LRU and MIN simulations disagree on route references or compulsory misses",
            ));
        }
        let layer = LayerOracleAnalysis {
            layer_index,
            physical_expert_slots: capacity,
            route_references: lru.route_references,
            unique_experts_touched,
            lru_cache_hits: lru.cache_hits,
            lru_missing_expert_installs: lru.missing_installs,
            lru_layer_miss_boundaries: lru.miss_boundaries,
            lru_bytes_to_move: checked_transfer_bytes(lru.missing_installs, slot_stride_bytes)?,
            minimum_cache_hits: minimum.cache_hits,
            minimum_missing_expert_installs: minimum.missing_installs,
            minimum_layer_miss_boundaries: minimum.miss_boundaries,
            minimum_bytes_to_move: checked_transfer_bytes(
                minimum.missing_installs,
                slot_stride_bytes,
            )?,
            compulsory_cold_installs: minimum.compulsory_cold_installs,
            lru_capacity_replacement_installs: lru
                .missing_installs
                .checked_sub(lru.compulsory_cold_installs)
                .ok_or_else(|| {
                    OracleTraceError::new(
                        "analysis-reconciliation",
                        "LRU compulsory misses exceed total misses",
                    )
                })?,
            minimum_capacity_replacement_installs: minimum
                .missing_installs
                .checked_sub(minimum.compulsory_cold_installs)
                .ok_or_else(|| {
                    OracleTraceError::new(
                        "analysis-reconciliation",
                        "MIN compulsory misses exceed total misses",
                    )
                })?,
            lru_resident_route_fraction: resident_fraction(lru.cache_hits, lru.route_references),
            maximum_potentially_resident_route_fraction: resident_fraction(
                minimum.cache_hits,
                minimum.route_references,
            ),
        };
        aggregate.physical_expert_slots += layer.physical_expert_slots;
        aggregate.route_references += layer.route_references;
        aggregate.unique_experts_touched += layer.unique_experts_touched;
        aggregate.lru_cache_hits += layer.lru_cache_hits;
        aggregate.lru_missing_expert_installs += layer.lru_missing_expert_installs;
        aggregate.lru_layer_miss_boundaries += layer.lru_layer_miss_boundaries;
        aggregate.lru_bytes_to_move += layer.lru_bytes_to_move;
        aggregate.minimum_cache_hits += layer.minimum_cache_hits;
        aggregate.minimum_missing_expert_installs += layer.minimum_missing_expert_installs;
        aggregate.minimum_layer_miss_boundaries += layer.minimum_layer_miss_boundaries;
        aggregate.minimum_bytes_to_move += layer.minimum_bytes_to_move;
        aggregate.compulsory_cold_installs += layer.compulsory_cold_installs;
        aggregate.lru_capacity_replacement_installs += layer.lru_capacity_replacement_installs;
        aggregate.minimum_capacity_replacement_installs +=
            layer.minimum_capacity_replacement_installs;
        per_layer.push(layer);
    }
    aggregate.lru_resident_route_fraction =
        resident_fraction(aggregate.lru_cache_hits, aggregate.route_references);
    aggregate.maximum_potentially_resident_route_fraction =
        resident_fraction(aggregate.minimum_cache_hits, aggregate.route_references);
    if aggregate.route_references != trace.total_selected_expert_ids as u64
        || aggregate.physical_expert_slots != plan.model_slot_capacity()
    {
        return Err(OracleTraceError::new(
            "analysis-reconciliation",
            "aggregate route references or slot capacity disagree with source evidence",
        ));
    }

    Ok(BudgetOracleAnalysis {
        requested_expert_budget_mib: requested_budget_mib,
        requested_expert_budget_bytes: plan.total_expert_budget_bytes(),
        minimum_executable_budget_bytes: plan.minimum_executable_budget_bytes(),
        arena_allocation_bytes: plan.total_arena_allocation_bytes(),
        unused_budget_bytes: plan.unused_remainder_bytes(),
        physical_slot_stride_bytes: slot_stride_bytes,
        physical_expert_slots: plan.model_slot_capacity(),
        per_layer_physical_expert_slots: capacities,
        per_layer,
        aggregate,
    })
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TraceRunEvidence {
    pub(crate) generated_token_ids: Vec<u32>,
    pub(crate) generated_token_ids_sha256: String,
    pub(crate) generated_text_sha256: String,
    pub(crate) route_sequence_sha256: String,
    pub(crate) legacy_selected_route_ids_sha256: String,
    pub(crate) token_loop: GpuNativeTokenLoopSnapshot,
    pub(crate) recovery: GpuNativeRecoverySnapshot,
    pub(crate) routed_execution: RoutedExpertExecutionSnapshot,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct DeterminismEvidence {
    pub(crate) identical_input: bool,
    pub(crate) greedy: bool,
    pub(crate) generated_token_ids_match: bool,
    pub(crate) generated_text_hashes_match: bool,
    pub(crate) ordered_route_sequence_hashes_match: bool,
    pub(crate) legacy_route_hashes_match: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct ObservationEvidence {
    pub(crate) ordinary_step_token_only: bool,
    pub(crate) seam: &'static str,
    pub(crate) additional_gpu_readback_introduced: bool,
    pub(crate) performance_claim_from_trace_run: bool,
    pub(crate) production_slot_planner: &'static str,
    pub(crate) analyzer_initial_residency: &'static str,
    pub(crate) minimum_policy: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct ExpertStorageEvidence {
    pub(crate) routed_expert_count: u64,
    pub(crate) expert_file_bytes: u64,
    pub(crate) routed_expert_footprint_bytes: u64,
    pub(crate) physical_slot_stride_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct OracleRouteTraceReport {
    pub(crate) schema: &'static str,
    pub(crate) mode: &'static str,
    pub(crate) complete: bool,
    pub(crate) diagnostic_only: bool,
    pub(crate) base_git_sha: &'static str,
    pub(crate) provenance: BenchmarkProvenance,
    pub(crate) model_identity: ModelIdentityEvidence,
    pub(crate) model_load: ModelLoadEvidence,
    pub(crate) adapter_identity: GpuDeviceIdentity,
    pub(crate) runtime_contract: RuntimeContractEvidence,
    pub(crate) request: RequestEvidence,
    pub(crate) production_configuration: ProductionConfiguration,
    pub(crate) expert_storage: ExpertStorageEvidence,
    pub(crate) observation: ObservationEvidence,
    pub(crate) warmup: TraceRunEvidence,
    pub(crate) measured: TraceRunEvidence,
    pub(crate) determinism: DeterminismEvidence,
    pub(crate) trace: OracleRouteTrace,
    pub(crate) offline_oracle_analysis: Vec<BudgetOracleAnalysis>,
    pub(crate) shutdown: BackgroundShutdownEvidence,
}

struct CapturedRun {
    evidence: TraceRunEvidence,
    trace: OracleRouteTrace,
}

fn counter_failure(error: impl std::fmt::Display) -> BenchmarkFailure {
    BenchmarkFailure::new(
        "postcondition",
        "trace-accounting-invalid",
        error.to_string(),
    )
}

async fn execute_trace_request(
    runtime: &crate::BenchRealRuntime,
    prompt_ids: &[u32],
    requested_output_tokens: usize,
    label: &str,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<CapturedRun, BenchmarkFailure> {
    let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
        BenchmarkFailure::new(
            "startup",
            "missing-gpu-native-token-loop",
            "oracle route trace runtime has no GPU-native token loop",
        )
    })?;
    let token_before = token_loop.snapshot();
    let recovery_before = token_loop.recovery_snapshot();
    let routed_before = runtime.engine.routed_expert_execution_snapshot();
    let collector = Arc::new(RouteCollector::default());
    runtime
        .engine
        .install_gpu_native_actual_route_observer(collector.clone())
        .map_err(|error| {
            BenchmarkFailure::new("startup", "route-observer-install-failed", error)
        })?;

    let execution = crate::with_progress_timeout(label.to_string(), watchdog, async {
        let mut request = token_loop.create_request_state()?;
        let mut generated_ids = Vec::with_capacity(requested_output_tokens);
        let mut completed_positions = 0usize;
        for (prompt_index, &token_id) in prompt_ids.iter().enumerate() {
            let sample = prompt_index + 1 == prompt_ids.len();
            let sampled = token_loop
                .step_token(
                    &runtime.engine,
                    &mut request,
                    token_id,
                    completed_positions,
                    sample,
                )
                .await?;
            completed_positions += 1;
            if sample {
                generated_ids.push(sampled.ok_or_else(|| {
                    BenchmarkFailure::new(
                        "inference",
                        "missing-first-generated-token",
                        "final prompt position produced no sampled token",
                    )
                })?);
            } else if sampled.is_some() {
                return Err(BenchmarkFailure::new(
                    "inference",
                    "unexpected-prefix-sample",
                    "prompt prefix position unexpectedly produced a sampled token",
                )
                .into());
            }
        }
        while generated_ids.len() < requested_output_tokens {
            let token_id = *generated_ids.last().ok_or_else(|| {
                BenchmarkFailure::new(
                    "inference",
                    "missing-generated-seed",
                    "decode has no preceding generated token",
                )
            })?;
            let sampled = token_loop
                .step_token(
                    &runtime.engine,
                    &mut request,
                    token_id,
                    completed_positions,
                    true,
                )
                .await?
                .ok_or_else(|| {
                    BenchmarkFailure::new(
                        "inference",
                        "missing-decode-token",
                        format!("position {completed_positions} produced no sampled token"),
                    )
                })?;
            generated_ids.push(sampled);
            completed_positions += 1;
        }
        Ok::<Vec<u32>, Box<dyn std::error::Error>>(generated_ids)
    })
    .await
    .map_err(|error| {
        BenchmarkFailure::new("inference", "route-trace-request-failed", error.to_string())
    });
    let clear = runtime.engine.clear_gpu_native_actual_route_observer();
    let generated_ids = match (execution, clear) {
        (Ok(ids), Ok(())) => ids,
        (Err(error), Ok(())) => return Err(error),
        (Ok(_), Err(clear_error)) => {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "route-observer-clear-failed",
                clear_error,
            ))
        }
        (Err(error), Err(clear_error)) => {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "route-request-and-observer-clear-failed",
                format!("{error}; {clear_error}"),
            ))
        }
    };

    let token_delta =
        crate::gpu_native_real_benchmark::token_loop_delta(token_before, token_loop.snapshot())?;
    let recovery_delta = crate::gpu_native_real_benchmark::recovery_delta(
        recovery_before,
        token_loop.recovery_snapshot(),
    )?;
    let routed_delta = crate::gpu_native_real_benchmark::routed_delta(
        routed_before,
        runtime.engine.routed_expert_execution_snapshot(),
    )?;
    crate::gpu_native_real_benchmark::validate_request_postconditions(
        prompt_ids.len(),
        requested_output_tokens,
        generated_ids.len(),
        token_delta,
        recovery_delta,
        routed_delta,
    )?;

    let trace = OracleRouteTrace::try_new(token_loop.model_geometry().into(), collector.take())
        .map_err(counter_failure)?;
    let trace_bytes = serde_json::to_vec(&trace).map_err(|error| {
        BenchmarkFailure::new(
            "postcondition",
            "trace-serialization-failed",
            error.to_string(),
        )
    })?;
    let trace = OracleRouteTrace::parse(&trace_bytes).map_err(counter_failure)?;
    if trace.total_positions as u64 != token_delta.tokens_completed
        || trace.total_selected_expert_ids as u64 != routed_delta.selected_routed_experts
    {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "trace-route-accounting-mismatch",
            format!(
                "trace positions={} selected_ids={} token_loop_completed={} routed_selected={}",
                trace.total_positions,
                trace.total_selected_expert_ids,
                token_delta.tokens_completed,
                routed_delta.selected_routed_experts
            ),
        ));
    }
    let output_text = runtime.tokenizer.decode(&generated_ids).map_err(|error| {
        BenchmarkFailure::new("postcondition", "output-decode-failed", error.to_string())
    })?;
    let evidence = TraceRunEvidence {
        generated_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&generated_ids),
        generated_text_sha256: crate::greedy_parity::sha256_hex(output_text.as_bytes()),
        route_sequence_sha256: trace.ordered_route_sequence_sha256.clone(),
        legacy_selected_route_ids_sha256: trace.legacy_selected_route_ids_sha256.clone(),
        generated_token_ids: generated_ids,
        token_loop: token_delta,
        recovery: recovery_delta,
        routed_execution: routed_delta,
    };
    Ok(CapturedRun { evidence, trace })
}

struct ValidatedRuntime {
    model_load: ModelLoadEvidence,
    adapter_identity: GpuDeviceIdentity,
    runtime_contract: RuntimeContractEvidence,
}

fn validate_runtime(
    runtime: &crate::BenchRealRuntime,
    expected_config_sha256: &str,
    expected_adapter_name: &str,
) -> Result<ValidatedRuntime, BenchmarkFailure> {
    let observed_config_sha256 = crate::resolved_real_runtime_identity_sha256(
        &runtime.cfg,
        runtime.model.config.architecture,
        runtime.model.config.first_k_dense_replace,
        &runtime.model.config.advanced,
    )
    .map_err(|error| {
        BenchmarkFailure::new(
            "startup",
            "runtime-config-identity-unavailable",
            error.to_string(),
        )
    })?;
    if observed_config_sha256 != expected_config_sha256 {
        return Err(BenchmarkFailure::new(
            "startup",
            "runtime-config-identity-drift",
            format!(
                "runtime identity {observed_config_sha256} differs from preflight identity {expected_config_sha256}"
            ),
        ));
    }
    if runtime.cfg.storage.predict_fanout != 0 {
        return Err(BenchmarkFailure::new(
            "startup",
            "prefetch-enabled",
            "ORACLE-0A requires storage.predict_fanout=0 at runtime",
        ));
    }
    let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
        BenchmarkFailure::new(
            "startup",
            "missing-gpu-native-token-loop",
            "oracle route trace runtime has no GPU-native token loop",
        )
    })?;
    if token_loop.snapshot() != GpuNativeTokenLoopSnapshot::default()
        || token_loop.recovery_snapshot() != GpuNativeRecoverySnapshot::default()
        || runtime.engine.routed_expert_execution_snapshot()
            != RoutedExpertExecutionSnapshot::default()
    {
        return Err(BenchmarkFailure::new(
            "startup",
            "nonzero-initial-runtime-counters",
            "fresh oracle route runtime did not begin with zero token/recovery/routed counters",
        ));
    }
    let model_load = crate::greedy_parity_model_load(runtime);
    let contract_input = RuntimeContractInput {
        real_transformer_enabled: runtime.cfg.real_transformer.enabled,
        real_transformer_gpu_native: runtime.cfg.real_transformer.gpu_native,
        compute_offload: runtime.cfg.real_transformer.compute_offload,
        legacy_execution_plan: runtime.engine.execution_context().plan().into(),
        token_loop_geometry: Some(token_loop.model_geometry()),
        authoritative_device: runtime.engine.gpu_device_identity(),
        model_load: model_load.clone(),
        routed_failure_policy: runtime.engine.routed_expert_gpu_failure_policy(),
    };
    let (runtime_contract, adapter_identity) =
        crate::gpu_native_real_benchmark::validate_runtime_contract(
            &contract_input,
            expected_adapter_name,
        )?;
    Ok(ValidatedRuntime {
        model_load,
        adapter_identity,
        runtime_contract,
    })
}

fn emit_report(
    report: &OracleRouteTraceReport,
    report_out: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = serde_json::to_vec_pretty(report)?;
    bytes.push(b'\n');
    if let Some(parent) = report_out
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(report_out, bytes)?;
    eprintln!(
        "GPU-native ORACLE-0A route trace written to {}",
        report_out.display()
    );
    Ok(())
}

pub(crate) async fn run_command(args: CommandArgs) -> Result<(), Box<dyn std::error::Error>> {
    let input = crate::load_real_cli_request_input(
        "trace-gpu-native-oracle-routes",
        None,
        Some(&args.request_json),
        None,
    )?;
    if input.output_tokens == 0 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "output-token-required",
            "ORACLE-0A requires at least one generated token",
        )
        .into());
    }

    let build = crate::qualification::BuildProvenance::embedded();
    crate::gpu_native_real_benchmark::validate_preflight_provenance(&build)?;
    let cfg = crate::config::Config::from_file(&args.config)?;
    crate::gpu_native_real_benchmark::validate_source_config(&cfg)?;
    if cfg.storage.predict_fanout != 0 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "prefetch-enabled",
            "ORACLE-0A requires frozen storage.predict_fanout=0",
        )
        .into());
    }
    let (artifacts, artifact_errors) = crate::qualification_artifacts(&args.config, &cfg);
    crate::gpu_native_real_benchmark::validate_artifacts(&artifacts, &artifact_errors)?;
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| {
                BenchmarkFailure::new("preflight", "expert-metadata-unavailable", error)
            })?;
    crate::gpu_native_real_benchmark::validate_expert_metadata(&expert_metadata)?;
    let spec = crate::resolve_real_cli_spec_from_config(
        cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
    )?;
    let model_identity = crate::greedy_parity_model_identity(&spec);
    if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "wrong-model-identity",
            format!(
                "requires exact Qwen3-Coder 30B-A3B Q4_0 identity; observed {model_identity:?}"
            ),
        )
        .into());
    }
    let routed_expert_count = u64::try_from(spec.cfg.model.num_layers)
        .map_err(|_| {
            BenchmarkFailure::new(
                "preflight",
                "expert-footprint-overflow",
                "model.num_layers does not fit u64",
            )
        })?
        .checked_mul(u64::from(spec.cfg.model.num_experts))
        .ok_or_else(|| {
            BenchmarkFailure::new(
                "preflight",
                "expert-footprint-overflow",
                "num_layers * num_experts overflowed",
            )
        })?;
    let expert_file_bytes = u64::try_from(spec.cfg.model.expert_size).map_err(|_| {
        BenchmarkFailure::new(
            "preflight",
            "expert-footprint-overflow",
            "model.expert_size does not fit u64",
        )
    })?;
    let routed_expert_footprint_bytes = routed_expert_count
        .checked_mul(expert_file_bytes)
        .ok_or_else(|| {
            BenchmarkFailure::new(
                "preflight",
                "expert-footprint-overflow",
                "routed expert count * model.expert_size overflowed",
            )
        })?;
    if spec.cfg.model.expert_size != EXPECTED_EXPERT_FILE_BYTES
        || routed_expert_footprint_bytes != EXPECTED_ROUTED_EXPERT_FOOTPRINT_BYTES
    {
        return Err(BenchmarkFailure::new(
            "preflight",
            "wrong-expert-footprint",
            format!(
                "requires {} routed experts of {EXPECTED_EXPERT_FILE_BYTES} bytes for an exact {EXPECTED_ROUTED_EXPERT_FOOTPRINT_BYTES}-byte footprint; observed {routed_expert_count} experts of {} bytes for {routed_expert_footprint_bytes} bytes",
                48 * 128,
                spec.cfg.model.expert_size,
            ),
        )
        .into());
    }
    let resolved_config_sha256 = crate::resolved_real_cli_spec_sha256(&spec)?;
    let tokenizer = crate::load_real_cli_tokenizer(
        &spec.cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
    )?;
    let prompt_ids = tokenizer.encode(&input.prompt)?;
    if prompt_ids.is_empty() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "empty-prompt-tokenization",
            "prompt encoded to zero tokens",
        )
        .into());
    }
    let (executable, executable_sha256) = crate::current_executable_identity()?;
    let executable_canonical_path = std::fs::canonicalize(&executable)?.display().to_string();
    if !crate::gpu_native_real_benchmark::is_hex(&executable_sha256, 64)
        || !crate::gpu_native_real_benchmark::is_hex(&resolved_config_sha256, 64)
    {
        return Err(BenchmarkFailure::new(
            "preflight",
            "provenance-unavailable",
            "executable or resolved-config SHA256 was unavailable",
        )
        .into());
    }
    let production_configuration =
        ProductionConfiguration::from_config(&spec.cfg, &expert_metadata);
    let provenance = BenchmarkProvenance {
        build,
        executable_canonical_path,
        executable_sha256,
        resolved_config_sha256: resolved_config_sha256.clone(),
        artifacts,
        expert_metadata,
    };
    let request = RequestEvidence {
        prompt_sha256: crate::greedy_parity::sha256_hex(input.prompt.as_bytes()),
        prompt_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&prompt_ids),
        prompt_token_count: prompt_ids.len(),
        requested_output_tokens: input.output_tokens,
        greedy: true,
    };

    let runtime = crate::build_isolated_greedy_runtime(
        &spec,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
        tokenizer,
    )
    .await?;
    let execution = async {
        let validated = validate_runtime(
            &runtime,
            &resolved_config_sha256,
            EXPECTED_ADAPTER_NAME,
        )?;
        let warmup = execute_trace_request(
            &runtime,
            &prompt_ids,
            input.output_tokens,
            "trace-gpu-native-oracle-routes warmup",
            args.progress_watchdog,
        )
        .await?;
        let measured = execute_trace_request(
            &runtime,
            &prompt_ids,
            input.output_tokens,
            "trace-gpu-native-oracle-routes measured",
            args.progress_watchdog,
        )
        .await?;
        let determinism = DeterminismEvidence {
            identical_input: true,
            greedy: true,
            generated_token_ids_match: warmup.evidence.generated_token_ids
                == measured.evidence.generated_token_ids,
            generated_text_hashes_match: warmup.evidence.generated_text_sha256
                == measured.evidence.generated_text_sha256,
            ordered_route_sequence_hashes_match: warmup.trace.ordered_route_sequence_sha256
                == measured.trace.ordered_route_sequence_sha256,
            legacy_route_hashes_match: warmup.trace.legacy_selected_route_ids_sha256
                == measured.trace.legacy_selected_route_ids_sha256,
        };
        if !determinism.generated_token_ids_match
            || !determinism.generated_text_hashes_match
            || !determinism.ordered_route_sequence_hashes_match
            || !determinism.legacy_route_hashes_match
        {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "nondeterministic-route-trace",
                "identical greedy warmup/measured requests produced different tokens, text, or routes",
            ));
        }

        let mut analysis = Vec::with_capacity(REQUESTED_BUDGETS_MIB.len());
        for budget_mib in REQUESTED_BUDGETS_MIB {
            let budget_bytes = budget_mib.checked_mul(MIB).ok_or_else(|| {
                BenchmarkFailure::new(
                    "postcondition",
                    "budget-overflow",
                    "requested analysis budget overflowed bytes",
                )
            })?;
            let plan = runtime
                .engine
                .gpu_native_model_expert_vram_plan_for_budget(budget_bytes)
                .map_err(|error| {
                    BenchmarkFailure::new(
                        "postcondition",
                        "slot-plan-failed",
                        error.to_string(),
                    )
                })?;
            analysis.push(
                analyze_trace_with_plan(&measured.trace, budget_mib, &plan)
                    .map_err(counter_failure)?,
            );
        }
        Ok::<_, BenchmarkFailure>((validated, warmup, measured, determinism, analysis))
    }
    .await;
    let shutdown = runtime.shutdown_isolated().await;
    let (validated, warmup, measured, determinism, analysis, shutdown) = match (execution, shutdown)
    {
        (Ok(values), Ok(shutdown)) => {
            if !shutdown.controlled_shutdown_requested || !shutdown.all_runtime_resources_released {
                return Err(BenchmarkFailure::new(
                    "postcondition",
                    "runtime-shutdown-incomplete",
                    format!("controlled isolated runtime shutdown was incomplete: {shutdown:?}"),
                )
                .into());
            }
            (values.0, values.1, values.2, values.3, values.4, shutdown)
        }
        (Err(error), Ok(_)) => return Err(error.into()),
        (Ok(_), Err(error)) => {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "runtime-shutdown-failed",
                error.to_string(),
            )
            .into())
        }
        (Err(execution_error), Err(shutdown_error)) => {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "execution-and-shutdown-failed",
                format!("{execution_error}; {shutdown_error}"),
            )
            .into())
        }
    };

    let report = OracleRouteTraceReport {
        schema: SCHEMA,
        mode: MODE,
        complete: true,
        diagnostic_only: true,
        base_git_sha: BASE_GIT_SHA,
        provenance,
        model_identity,
        model_load: validated.model_load,
        adapter_identity: validated.adapter_identity,
        runtime_contract: validated.runtime_contract,
        request,
        production_configuration,
        expert_storage: ExpertStorageEvidence {
            routed_expert_count,
            expert_file_bytes,
            routed_expert_footprint_bytes,
            physical_slot_stride_bytes: analysis
                .first()
                .ok_or_else(|| {
                    BenchmarkFailure::new(
                        "postcondition",
                        "missing-offline-analysis",
                        "no requested VRAM budget analysis was produced",
                    )
                })?
                .physical_slot_stride_bytes,
        },
        observation: ObservationEvidence {
            ordinary_step_token_only: true,
            seam: "GpuNativeTokenLoop::step_token -> step_token_unified_inner completion -> Engine::record_gpu_native_actual_routes(position, boundary_report.selected_ids)",
            additional_gpu_readback_introduced: false,
            performance_claim_from_trace_run: false,
            production_slot_planner: "GpuNativeModelExpertVramPlan::try_new with authoritative GpuNativeExecutorContext::device_limits",
            analyzer_initial_residency: "empty per-layer physical arenas",
            minimum_policy: "simultaneous-set Belady/MIN; current route protected, farthest next-use unprotected resident evicted",
        },
        warmup: warmup.evidence,
        measured: measured.evidence,
        determinism,
        trace: measured.trace,
        offline_oracle_analysis: analysis,
        shutdown,
    };
    emit_report(&report, &args.report_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(layers: usize, experts: usize, top_k: usize) -> OracleGeometry {
        OracleGeometry {
            layers,
            experts,
            top_k,
            d_model: 32,
            d_ff: 32,
        }
    }

    fn small_trace() -> OracleRouteTrace {
        OracleRouteTrace::try_new(
            geometry(2, 4, 2),
            vec![
                OracleRouteRecord {
                    position: 0,
                    layer_index: 0,
                    ordered_selected_expert_ids: vec![2, 1],
                },
                OracleRouteRecord {
                    position: 0,
                    layer_index: 1,
                    ordered_selected_expert_ids: vec![0, 3],
                },
                OracleRouteRecord {
                    position: 1,
                    layer_index: 0,
                    ordered_selected_expert_ids: vec![1, 3],
                },
                OracleRouteRecord {
                    position: 1,
                    layer_index: 1,
                    ordered_selected_expert_ids: vec![2, 0],
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn canonical_route_serialization_and_hashing_are_order_sensitive() {
        let trace = small_trace();
        let bytes = canonical_route_bytes(&trace.records).unwrap();
        assert!(bytes.starts_with(b"mer.gpu-native-oracle-route-sequence.v1\0"));
        assert_eq!(
            trace.ordered_route_sequence_sha256,
            ordered_route_sequence_sha256(&trace.records).unwrap()
        );
        let mut reordered = trace.records.clone();
        reordered[0].ordered_selected_expert_ids.swap(0, 1);
        assert_ne!(
            trace.ordered_route_sequence_sha256,
            ordered_route_sequence_sha256(&reordered).unwrap()
        );
    }

    #[test]
    fn valid_trace_reconciles_counts_and_hashes() {
        let trace = small_trace();
        trace.validate().unwrap();
        assert_eq!(trace.total_positions, 2);
        assert_eq!(trace.total_layer_records, 4);
        assert_eq!(trace.total_selected_expert_ids, 8);
        assert_ne!(
            trace.ordered_route_sequence_sha256,
            trace.legacy_selected_route_ids_sha256
        );
    }

    #[test]
    fn invalid_expert_id_is_rejected() {
        let mut trace = small_trace();
        trace.records[0].ordered_selected_expert_ids[0] = 4;
        assert_eq!(trace.validate().unwrap_err().code, "invalid-expert-id");
    }

    #[test]
    fn duplicate_top_k_id_is_rejected() {
        let mut trace = small_trace();
        trace.records[0].ordered_selected_expert_ids = vec![1, 1];
        assert_eq!(trace.validate().unwrap_err().code, "duplicate-top-k-id");
    }

    #[test]
    fn wrong_layer_count_is_rejected() {
        let mut trace = small_trace();
        trace.records.pop();
        assert_eq!(trace.validate().unwrap_err().code, "wrong-layer-count");
    }

    #[test]
    fn wrong_top_k_count_is_rejected() {
        let mut trace = small_trace();
        trace.records[0].ordered_selected_expert_ids.pop();
        assert_eq!(trace.validate().unwrap_err().code, "wrong-top-k-count");
    }

    #[test]
    fn trace_json_parsing_is_deterministic_and_validated() {
        let trace = small_trace();
        let bytes = serde_json::to_vec(&trace).unwrap();
        let first = OracleRouteTrace::parse(&bytes).unwrap();
        let second = OracleRouteTrace::parse(&bytes).unwrap();
        assert_eq!(first, second);
        let mut corrupt = serde_json::to_value(&trace).unwrap();
        corrupt["records"][0]["ordered_selected_expert_ids"][0] = serde_json::json!(99);
        assert_eq!(
            OracleRouteTrace::parse(&serde_json::to_vec(&corrupt).unwrap())
                .unwrap_err()
                .code,
            "invalid-expert-id"
        );
    }

    #[test]
    fn belady_min_is_optimal_on_small_single_access_sequence() {
        let accesses = vec![vec![0], vec![1], vec![2], vec![0], vec![1], vec![2]];
        let lru = simulate_production_lru(2, 3, &accesses).unwrap();
        let min = simulate_belady_min(2, 3, &accesses).unwrap();
        assert_eq!(lru.missing_installs, 6);
        assert_eq!(min.missing_installs, 4);
        assert_eq!(min.compulsory_cold_installs, 3);
        assert_eq!(min.miss_boundaries, 4);
    }

    #[test]
    fn simultaneous_top_k_access_protects_the_entire_current_set() {
        let accesses = vec![vec![0, 1], vec![1, 2], vec![0, 2]];
        let min = simulate_belady_min(2, 3, &accesses).unwrap();
        assert_eq!(min.route_references, 6);
        assert_eq!(min.cache_hits, 2);
        assert_eq!(min.missing_installs, 4);
        assert_eq!(min.miss_boundaries, 3);
    }

    #[test]
    fn full_capacity_has_only_compulsory_cold_installs() {
        let accesses = vec![vec![0, 1], vec![2, 3], vec![0, 3], vec![1, 2]];
        let lru = simulate_production_lru(4, 4, &accesses).unwrap();
        let min = simulate_belady_min(4, 4, &accesses).unwrap();
        assert_eq!(lru.missing_installs, 4);
        assert_eq!(min.missing_installs, 4);
        assert_eq!(min.compulsory_cold_installs, 4);
        assert_eq!(min.missing_installs - min.compulsory_cold_installs, 0);
    }

    #[test]
    fn capacity_constrained_set_sequence_reports_replacement_misses() {
        let accesses = vec![vec![0, 1], vec![2, 3], vec![0, 1], vec![2, 3]];
        let min = simulate_belady_min(2, 4, &accesses).unwrap();
        assert_eq!(min.compulsory_cold_installs, 4);
        assert_eq!(min.missing_installs, 8);
        assert_eq!(min.miss_boundaries, 4);
    }

    #[test]
    fn production_model_slot_planner_reuse_matches_qwen_geometry() {
        let qwen = GpuNativeQ4ExpertGeometry::try_new(2048, 768, 128, 8).unwrap();
        let budget = 15_576 * MIB;
        let limits = wgpu::Limits {
            max_push_constant_size: 32,
            max_storage_buffers_per_shader_stage: 8,
            max_compute_workgroup_size_x: 64,
            max_compute_invocations_per_workgroup: 64,
            ..wgpu::Limits::default()
        };
        let plan = GpuNativeModelExpertVramPlan::try_new(48, qwen, budget, &limits).unwrap();
        assert_eq!(plan.num_layers(), 48);
        assert_eq!(plan.model_slot_capacity(), 48 * 128);
        assert!(plan
            .layer_plans()
            .iter()
            .all(|layer| layer.slot_capacity() == 128));
        assert!(plan.total_arena_allocation_bytes() <= budget);
        assert_eq!(qwen.slot_stride_bytes(), 2_654_212);
    }
}
