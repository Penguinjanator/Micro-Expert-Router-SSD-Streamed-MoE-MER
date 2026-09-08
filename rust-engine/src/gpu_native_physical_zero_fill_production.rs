//! Same-child, same-concurrency qualification of production zero-fill elision.
//! Reuses physical-install workload, isolated runtime, observation and teardown.
use super::*;
use crate::engine::{
    GpuNativePhysicalInstallConcurrencyQualificationArm as Arm,
    GpuNativePhysicalInstallConcurrencyQualificationSnapshot as Snapshot,
};

pub(crate) const SCHEMA: &str = "mer.gpu-native-physical-zero-fill-production.v1";
pub(crate) const MODE: &str = "qualify-gpu-native-physical-zero-fill-production";
const LOGICAL_EXPERT_BYTES: u64 = 2_654_208;
const SLOT_STRIDE_BYTES: u64 = 2_654_212;
const EPOCH_BYTES: u64 = 4;

const fn qualification_arms() -> (Arm, Arm) {
    (
        Arm::ConcurrentFullZeroControl,
        Arm::ProductionNoZeroFillTreatment,
    )
}

#[derive(Clone, Debug, Default, Serialize)]
struct PairMechanismGate {
    explicit_full_zero_control: bool,
    ordinary_no_zero_treatment: bool,
    both_direct_no_vec: bool,
    install_accounting_exact: bool,
    full_zero_bytes_equal_staged: bool,
    no_zero_bytes_zero: bool,
    epoch_payload_staged_bytes_exact: bool,
    concurrent_set_widths_and_behavior_exact: bool,
    physical_install_and_victim_order_exact: bool,
    source_and_route_streams_exact: bool,
    failures_and_accounting_errors_zero: bool,
    timing_accounting_exact: bool,
    passed: bool,
}

pub(super) fn install_accounting_exact(
    s: &Snapshot,
    p: &GpuNativeProductionPhysicalInstallSnapshot,
) -> bool {
    let installs = s.physical_install_completions;
    installs > 0
        && production_concurrency_exercised(p)
        && s.physical_install_attempts == installs
        && s.physical_install_experts == installs
        && s.reservation_attempts == installs
        && s.reservation_successes == installs
        && s.physical_stage_attempts == installs
        && s.physical_stage_completions == installs
        && s.direct_staging_writes == installs
        && s.ordered_commit_attempts == installs
        && s.ordered_commit_completions == installs
        && s.mapping_publications == installs
        && s.physical_install_sets == p.physical_install_sets
        && installs == p.physical_install_experts
        && installs == p.physical_install_attempts
        && installs == p.physical_stage_completions
        && installs == p.ordered_commit_completions
        && s.parallel_eligible_sets == p.parallel_eligible_sets
        && s.parallel_staging_sets == p.parallel_staging_sets
        && s.parallel_staging_experts == p.parallel_staging_experts
        && s.singleton_staging_sets == p.singleton_staging_sets
        && s.parallel_staging_sets > 0
        && s.parallel_staging_sets == s.parallel_eligible_sets
        && s.parallel_staging_experts == s.parallel_eligible_experts
        && s.parallel_staging_experts
            .checked_add(s.singleton_staging_sets)
            == Some(installs)
        && s.parallel_staging_sets
            .checked_add(s.singleton_staging_sets)
            == Some(s.physical_install_sets)
        && s.max_in_flight_physical_staging >= 2
}

pub(super) fn bytes_exact(s: &Snapshot) -> bool {
    let installs = s.physical_install_completions;
    installs.checked_mul(EPOCH_BYTES) == Some(s.physical_slot_epoch_write_bytes)
        && installs.checked_mul(LOGICAL_EXPERT_BYTES) == Some(s.physical_slot_payload_copy_bytes)
        && installs.checked_mul(SLOT_STRIDE_BYTES) == Some(s.physical_slot_bytes_staged)
        && s.physical_bytes_staged == s.physical_slot_bytes_staged
}

pub(super) fn failures_zero(s: &Snapshot) -> bool {
    s.reservation_failures == 0
        && s.direct_staging_failures == 0
        && s.physical_stage_failures == 0
        && s.ordered_commit_failures == 0
        && s.ordered_commit_violations == 0
        && s.unpublished_physical_writes_after_failure == 0
        && s.active_physical_staging == 0
        && s.evidence_accounting_errors == 0
        && s.timing_accounting_errors == 0
        && s.overlapping_demand_sets == 0
        && s.single_request_stream
}

pub(super) fn timing_accounting_exact(s: &Snapshot) -> bool {
    s.sum_individual_physical_stage_us
        .checked_add(s.physical_ordered_commit_us)
        == Some(s.physical_install_total_us)
        && s.physical_slot_prepare_us
            .checked_add(s.physical_queue_staging_us)
            .is_some_and(|parts| parts <= s.sum_individual_physical_stage_us)
        && s.mapping_publication_us <= s.physical_ordered_commit_us
}

fn pair_mechanism_gate(
    c: &Snapshot,
    t: &Snapshot,
    cp: &GpuNativeProductionPhysicalInstallSnapshot,
    tp: &GpuNativeProductionPhysicalInstallSnapshot,
) -> PairMechanismGate {
    let mut gate = PairMechanismGate {
        explicit_full_zero_control: c.arm == qualification_arms().0
            && !c.production_physical_install_concurrency_changed
            && !c.control_forces_sequential_direct_staging
            && !c.treatment_uses_ordinary_production_path,
        ordinary_no_zero_treatment: t.arm == qualification_arms().1
            && !t.production_physical_install_concurrency_changed
            && t.treatment_uses_ordinary_production_path
            && !t.control_forces_sequential_direct_staging,
        both_direct_no_vec: c.direct_staging_writes > 0
            && t.direct_staging_writes > 0
            && c.full_slot_vec_materializations == 0
            && t.full_slot_vec_materializations == 0,
        install_accounting_exact: install_accounting_exact(c, cp)
            && install_accounting_exact(t, tp)
            && c.physical_install_completions == t.physical_install_completions,
        full_zero_bytes_equal_staged: c.physical_slot_zero_fill_bytes > 0
            && c.physical_slot_zero_fill_bytes == c.physical_slot_bytes_staged,
        no_zero_bytes_zero: t.physical_slot_zero_fill_bytes == 0,
        epoch_payload_staged_bytes_exact: bytes_exact(c)
            && bytes_exact(t)
            && c.physical_slot_epoch_write_bytes == t.physical_slot_epoch_write_bytes
            && c.physical_slot_payload_copy_bytes == t.physical_slot_payload_copy_bytes
            && c.physical_slot_bytes_staged == t.physical_slot_bytes_staged,
        concurrent_set_widths_and_behavior_exact: c
            .normal_production_uses_concurrent_physical_staging
            && t.normal_production_uses_concurrent_physical_staging
            && c.physical_install_sets == t.physical_install_sets
            && c.install_set_width_min == t.install_set_width_min
            && c.install_set_width_max == t.install_set_width_max
            && c.install_set_width_mean == t.install_set_width_mean
            && c.parallel_eligible_sets == t.parallel_eligible_sets
            && c.parallel_eligible_experts == t.parallel_eligible_experts
            && c.parallel_staging_sets == t.parallel_staging_sets
            && c.parallel_staging_experts == t.parallel_staging_experts
            && c.singleton_staging_sets == t.singleton_staging_sets
            && c.rayon_num_threads >= 2
            && c.rayon_num_threads == t.rayon_num_threads
            && c.caller_was_already_rayon_worker == t.caller_was_already_rayon_worker
            && c.ordered_install_set_behavior_sha256 == t.ordered_install_set_behavior_sha256,
        physical_install_and_victim_order_exact: c.physical_victim_ids_sha256
            == t.physical_victim_ids_sha256
            && c.reservation_identity_sha256 == t.reservation_identity_sha256
            && c.physical_residency_identity_sha256 == t.physical_residency_identity_sha256
            && c.mapping_publications == t.mapping_publications
            && c.mapping_unpublications == t.mapping_unpublications,
        source_and_route_streams_exact: c.selected_route_ids_sha256 == t.selected_route_ids_sha256
            && c.physical_missing_ids_sha256 == t.physical_missing_ids_sha256
            && c.demand_source_request_ids_sha256 == t.demand_source_request_ids_sha256
            && c.demand_source_requests == t.demand_source_requests
            && c.source_nvme_reads == t.source_nvme_reads
            && c.source_nvme_bytes == t.source_nvme_bytes,
        failures_and_accounting_errors_zero: failures_zero(c) && failures_zero(t),
        timing_accounting_exact: timing_accounting_exact(c) && timing_accounting_exact(t),
        passed: false,
    };
    gate.passed = gate.explicit_full_zero_control
        && gate.ordinary_no_zero_treatment
        && gate.both_direct_no_vec
        && gate.install_accounting_exact
        && gate.full_zero_bytes_equal_staged
        && gate.no_zero_bytes_zero
        && gate.epoch_payload_staged_bytes_exact
        && gate.concurrent_set_widths_and_behavior_exact
        && gate.physical_install_and_victim_order_exact
        && gate.source_and_route_streams_exact
        && gate.failures_and_accounting_errors_zero
        && gate.timing_accounting_exact;
    gate
}

#[derive(Clone, Debug, Serialize)]
struct Gates {
    behavioral: BehavioralGate,
    work_equivalence: WorkEquivalenceGate,
    warmup_mechanism: PairMechanismGate,
    measured_mechanism: PairMechanismGate,
    warmup_and_measured_work_exact: bool,
    token_ids_text_and_routes_exact: bool,
    stale_generation_and_install_errors_zero: bool,
    physical_installs_reconcile_with_residency: bool,
    passed: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Performance {
    #[serde(flatten)]
    physical_install: ConcurrencyPerformanceComparison,
    mean_time_to_first_token_seconds: MetricComparison,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    schema: &'static str,
    mode: &'static str,
    control: Option<ConcurrencyArmReport>,
    treatment: Option<ConcurrencyArmReport>,
    control_path: &'static str,
    treatment_path: &'static str,
    both_arms_concurrent_direct_staging: bool,
    source_scheduler_changed: bool,
    staging_byte_count_changed: bool,
    queue_ordering_changed: bool,
    payload_offset_bytes: u64,
    logical_expert_bytes: u64,
    slot_stride_bytes: u64,
    tail_padding_bytes: u64,
    no_zero_requires_complete_coverage_before_first_write: bool,
    frozen_workload: FrozenWorkload,
    provenance: BenchmarkProvenance,
    timing_definitions: ConcurrencyTimingDefinitions,
    reconciliation: Option<ProductionReconciliation>,
    gates: Option<Gates>,
    performance: Option<Performance>,
    benchmark_complete: bool,
    qualification_pass: bool,
    performance_result: &'static str,
    failure: Option<BenchmarkFailure>,
}

fn work_pair_exact(c: &ArmWorkEvidence, t: &ArmWorkEvidence) -> bool {
    c.gpu_native_residency
        .logical_admissions_for_physical_misses
        == t.gpu_native_residency
            .logical_admissions_for_physical_misses
        && c.gpu_native_residency.ram_to_vram_installs
            == t.gpu_native_residency.ram_to_vram_installs
        && c.gpu_native_residency.physical_evictions == t.gpu_native_residency.physical_evictions
        && c.gpu_native_residency.physical_reinstalls == t.gpu_native_residency.physical_reinstalls
        && c.token_loop.residency_miss_attempts == t.token_loop.residency_miss_attempts
        && c.token_loop.residency_services == t.token_loop.residency_services
        && recovery_semantics_equal(c.recovery, t.recovery)
        && c.routed_execution.selected_routed_experts == t.routed_execution.selected_routed_experts
        && c.engine_storage.ram_hits == t.engine_storage.ram_hits
        && c.engine_storage.ram_misses == t.engine_storage.ram_misses
        && c.engine_storage.nvme_read_operations == t.engine_storage.nvme_read_operations
        && c.engine_storage.nvme_bytes_read == t.engine_storage.nvme_bytes_read
        && c.token_loop.queue_submissions == t.token_loop.queue_submissions
        && c.token_loop.boundary_maps == t.token_loop.boundary_maps
        && c.token_loop.boundary_readbacks == t.token_loop.boundary_readbacks
}

pub(super) fn work_errors_zero(w: &ArmWorkEvidence) -> bool {
    w.gpu_native_residency.stale_generation_rejections == 0
        && w.token_loop.fatal_failures == 0
        && w.token_loop.no_progress_failures == 0
        && w.token_loop.replay_attempts == 0
        && w.recovery.full_token_replay_attempts == 0
}

// Performance has no argument and cannot affect correctness/mechanism PASS.
fn qualification_pass(reconciliation_pass: bool, gates: &Gates) -> bool {
    reconciliation_pass && gates.passed
}

pub(crate) async fn run_command(args: CommandArgs) -> Result<(), Box<dyn std::error::Error>> {
    let prepared = prepare(&args)?;
    let mut timing_definitions = concurrency_timing_definitions();
    timing_definitions.common.physical_install_total_us = "both arms: sum of each post-reservation physical stage plus ordered commit service; excludes reservation time";
    timing_definitions.common.physical_slot_prepare_us = "both arms: validated host preparation; control zeros the full slot, treatment completely overwrites epoch and payload without explicit zero fill";
    timing_definitions.common.mapping_publication_us =
        "both arms: ordered logical mapping Queue::write_buffer time after all staging jobs finish";
    let mut report = Report {
        schema: SCHEMA,
        mode: MODE,
        control: None,
        treatment: None,
        control_path: "concurrent direct staging with explicit full-slot zero fill",
        treatment_path:
            "ordinary production concurrent direct staging with complete-overwrite no-zero fill",
        both_arms_concurrent_direct_staging: true,
        source_scheduler_changed: false,
        staging_byte_count_changed: false,
        queue_ordering_changed: false,
        payload_offset_bytes: EPOCH_BYTES,
        logical_expert_bytes: LOGICAL_EXPERT_BYTES,
        slot_stride_bytes: SLOT_STRIDE_BYTES,
        tail_padding_bytes: 0,
        no_zero_requires_complete_coverage_before_first_write: true,
        frozen_workload: frozen_workload(args.expected_adapter_name.clone()),
        provenance: prepared.provenance.clone(),
        timing_definitions,
        reconciliation: None,
        gates: None,
        performance: None,
        benchmark_complete: false,
        qualification_pass: false,
        performance_result: "not_measured",
        failure: None,
    };
    let (control_arm, treatment_arm) = qualification_arms();
    for arm in [control_arm, treatment_arm] {
        let result = run_physical_install_arm(
            &prepared,
            &args,
            PhysicalInstallQualificationRun::ZeroFillProduction(arm),
        )
        .await;
        let run = match result {
            Ok(run) => run,
            Err(failure) => {
                report.failure = Some(failure.clone());
                emit_report(&report, &args.report_out)?;
                return Err(failure.to_string().into());
            }
        };
        let failure = run.common.failure.clone();
        let arm_report = ConcurrencyArmReport {
            common: run.common,
            warmup_mechanism: run.warmup_concurrency,
            mechanism: run.concurrency,
        };
        let snapshots_present = arm_report.common.complete
            && arm_report.mechanism.is_some()
            && arm_report.warmup_mechanism.is_some()
            && arm_report.common.source.is_some()
            && arm_report.common.warmup_source.is_some()
            && arm_report.common.work.is_some()
            && arm_report.common.warmup_work.is_some()
            && arm_report.common.production.is_some()
            && arm_report.common.warmup_production.is_some()
            && arm_report.common.production_physical_install.is_some()
            && arm_report
                .common
                .warmup_production_physical_install
                .is_some();
        if arm == control_arm {
            report.control = Some(arm_report);
        } else {
            report.treatment = Some(arm_report);
        }
        if let Some(failure) = failure.or_else(|| {
            (!snapshots_present).then(|| {
                BenchmarkFailure::new(
                    "postcondition",
                    "missing-zero-fill-arm-evidence",
                    "complete warmup and measured evidence is required",
                )
            })
        }) {
            report.failure = Some(failure.clone());
            emit_report(&report, &args.report_out)?;
            return Err(failure.to_string().into());
        }
    }
    let c = report.control.as_ref().expect("stored control");
    let t = report.treatment.as_ref().expect("stored treatment");
    let reconciliation =
        production_reconciliation(reconcile(&c.common, &t.common), &c.common, &t.common);
    let (behavioral, work_equivalence) = common_gates(&reconciliation.common);
    let warmup_mechanism = pair_mechanism_gate(
        c.warmup_mechanism.as_ref().unwrap(),
        t.warmup_mechanism.as_ref().unwrap(),
        c.common
            .warmup_production_physical_install
            .as_ref()
            .unwrap(),
        t.common
            .warmup_production_physical_install
            .as_ref()
            .unwrap(),
    );
    let measured_mechanism = pair_mechanism_gate(
        c.mechanism.as_ref().unwrap(),
        t.mechanism.as_ref().unwrap(),
        c.common.production_physical_install.as_ref().unwrap(),
        t.common.production_physical_install.as_ref().unwrap(),
    );
    let cw = c.common.warmup_work.as_ref().unwrap();
    let tw = t.common.warmup_work.as_ref().unwrap();
    let cm = c.common.work.as_ref().unwrap();
    let tm = t.common.work.as_ref().unwrap();
    let warmup_and_measured_work_exact = work_pair_exact(cw, tw) && work_pair_exact(cm, tm);
    let stale_generation_and_install_errors_zero =
        [cw, tw, cm, tm].into_iter().all(work_errors_zero);
    let physical_installs_reconcile_with_residency = [
        (cw, c.warmup_mechanism.as_ref().unwrap()),
        (tw, t.warmup_mechanism.as_ref().unwrap()),
        (cm, c.mechanism.as_ref().unwrap()),
        (tm, t.mechanism.as_ref().unwrap()),
    ]
    .into_iter()
    .all(|(work, mechanism)| {
        work.gpu_native_residency.ram_to_vram_installs == mechanism.physical_install_completions
    });
    let token_ids_text_and_routes_exact = c.common.warmup_results.len() == FROZEN_WARMUP_RUNS
        && t.common.warmup_results.len() == FROZEN_WARMUP_RUNS
        && generated_results(&c.common).len() == FROZEN_MEASURED_RUNS
        && generated_results(&t.common).len() == FROZEN_MEASURED_RUNS
        && c.common
            .warmup_results
            .iter()
            .zip(&t.common.warmup_results)
            .all(|(a, b)| {
                a.generated_tokens == FROZEN_OUTPUT_TOKENS
                    && b.generated_tokens == FROZEN_OUTPUT_TOKENS
                    && a.generated_text_sha256 == b.generated_text_sha256
                    && a.generated_token_ids_sha256 == b.generated_token_ids_sha256
            })
        && generated_results(&c.common)
            .iter()
            .zip(generated_results(&t.common))
            .all(|(a, b)| {
                a.generated_tokens == FROZEN_OUTPUT_TOKENS
                    && b.generated_tokens == FROZEN_OUTPUT_TOKENS
                    && a.generated_text_sha256 == b.generated_text_sha256
                    && a.generated_token_ids_sha256 == b.generated_token_ids_sha256
            })
        && warmup_mechanism.source_and_route_streams_exact
        && measured_mechanism.source_and_route_streams_exact;
    let passed = behavioral.passed
        && work_equivalence.passed
        && warmup_mechanism.passed
        && measured_mechanism.passed
        && warmup_and_measured_work_exact
        && token_ids_text_and_routes_exact
        && stale_generation_and_install_errors_zero
        && physical_installs_reconcile_with_residency;
    let gates = Gates {
        behavioral,
        work_equivalence,
        warmup_mechanism,
        measured_mechanism,
        warmup_and_measured_work_exact,
        token_ids_text_and_routes_exact,
        stale_generation_and_install_errors_zero,
        physical_installs_reconcile_with_residency,
        passed,
    };
    let performance = concurrency_performance(c, t).map(|physical_install| Performance {
        physical_install,
        mean_time_to_first_token_seconds: comparison(
            generated_results(&c.common)
                .iter()
                .map(|r| r.timing.time_to_first_token_seconds)
                .sum::<f64>()
                / FROZEN_MEASURED_RUNS as f64,
            generated_results(&t.common)
                .iter()
                .map(|r| r.timing.time_to_first_token_seconds)
                .sum::<f64>()
                / FROZEN_MEASURED_RUNS as f64,
        ),
    });
    report.qualification_pass = qualification_pass(reconciliation.all_invariants_pass, &gates);
    report.reconciliation = Some(reconciliation);
    report.gates = Some(gates);
    match performance {
        Ok(performance) => {
            report.benchmark_complete = true;
            report.performance_result = performance.physical_install.common.performance_result;
            report.performance = Some(performance);
        }
        Err(failure) => {
            report.qualification_pass = false;
            report.failure = Some(failure.clone());
            emit_report(&report, &args.report_out)?;
            return Err(failure.to_string().into());
        }
    }
    emit_report(&report, &args.report_out)?;
    if report.qualification_pass {
        Ok(())
    } else {
        Err("physical zero-fill production qualification gates did not all pass; see emitted report".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(arm: Arm) -> (Snapshot, GpuNativeProductionPhysicalInstallSnapshot) {
        let mut s = crate::engine::empty_physical_zero_fill_test_snapshot(arm);
        s.physical_install_attempts = 3;
        s.physical_install_completions = 3;
        s.physical_install_experts = 3;
        s.direct_staging_writes = 3;
        s.reservation_attempts = 3;
        s.reservation_successes = 3;
        s.physical_stage_attempts = 3;
        s.physical_stage_completions = 3;
        s.ordered_commit_attempts = 3;
        s.ordered_commit_completions = 3;
        s.mapping_publications = 3;
        s.physical_slot_bytes_staged = 3 * SLOT_STRIDE_BYTES;
        s.physical_bytes_staged = s.physical_slot_bytes_staged;
        s.physical_slot_zero_fill_bytes = if arm == Arm::ProductionNoZeroFillTreatment {
            0
        } else {
            s.physical_slot_bytes_staged
        };
        s.physical_slot_epoch_write_bytes = 3 * EPOCH_BYTES;
        s.physical_slot_payload_copy_bytes = 3 * LOGICAL_EXPERT_BYTES;
        s.physical_install_sets = 1;
        s.install_set_width_min = 3;
        s.install_set_width_max = 3;
        s.install_set_width_mean = 3.0;
        s.parallel_eligible_sets = 1;
        s.parallel_eligible_experts = 3;
        s.parallel_staging_sets = 1;
        s.parallel_staging_experts = 3;
        s.max_in_flight_physical_staging = 3;
        s.rayon_num_threads = 4;
        s.physical_slot_prepare_us = 10;
        s.physical_queue_staging_us = 5;
        s.sum_individual_physical_stage_us = 30;
        s.physical_ordered_commit_us = 9;
        s.mapping_publication_us = 3;
        s.physical_install_total_us = 39;
        let p = GpuNativeProductionPhysicalInstallSnapshot {
            physical_install_sets: 1,
            physical_install_experts: 3,
            parallel_eligible_sets: 1,
            parallel_staging_sets: 1,
            parallel_staging_experts: 3,
            reservation_attempts: 3,
            reservation_successes: 3,
            physical_install_attempts: 3,
            physical_stage_attempts: 3,
            physical_stage_completions: 3,
            direct_staging_successes: 3,
            ordered_commit_attempts: 3,
            ordered_commit_completions: 3,
            max_in_flight_physical_staging: 3,
            ..GpuNativeProductionPhysicalInstallSnapshot::default()
        };
        (s, p)
    }

    #[test]
    fn zero_fill_production_frozen_contract_selects_only_concurrent_control_and_production() {
        assert_eq!(SCHEMA, "mer.gpu-native-physical-zero-fill-production.v1");
        assert_eq!(MODE, "qualify-gpu-native-physical-zero-fill-production");
        assert_ne!(SCHEMA, CONCURRENCY_SCHEMA);
        assert_eq!(
            qualification_arms(),
            (
                Arm::ConcurrentFullZeroControl,
                Arm::ProductionNoZeroFillTreatment
            )
        );
        assert_eq!(
            FROZEN_PROMPT,
            "Write a Rust function that adds two i32 values and returns the result."
        );
        assert_eq!(
            (
                FROZEN_OUTPUT_TOKENS,
                FROZEN_WARMUP_RUNS,
                FROZEN_MEASURED_RUNS
            ),
            (128, 1, 3)
        );
        assert_eq!(
            FROZEN_CONFIG_PATH,
            "/home/randyap8/slice11-qwen3-coder-gpu-native.toml"
        );
        assert_eq!(
            FROZEN_CONFIG_SHA256,
            "33d7cf96328d9c68b0ff45448d91d597d2e3a757cb99e6e61c72998ceabdd056"
        );
        let workload = frozen_workload("NVIDIA L4".into());
        assert_eq!(workload.cache_reset, "keep");
        assert_eq!(workload.sampling, "greedy");
        assert_eq!(EPOCH_BYTES + LOGICAL_EXPERT_BYTES, SLOT_STRIDE_BYTES);
    }

    #[test]
    fn zero_fill_production_accepts_exact_bytes_and_concurrent_identity() {
        let (c, cp) = fixture(qualification_arms().0);
        let (t, tp) = fixture(qualification_arms().1);
        let gate = pair_mechanism_gate(&c, &t, &cp, &tp);
        assert!(gate.passed, "{gate:?}");
        // Timing magnitudes and performance gains cannot rescue or defeat a
        // valid mechanism; timing decomposition still must reconcile.
        let mut slow = t.clone();
        slow.sum_individual_physical_stage_us = 300_000;
        slow.physical_install_total_us = 300_009;
        assert!(pair_mechanism_gate(&c, &slow, &cp, &tp).passed);
    }

    #[test]
    fn zero_fill_production_rejects_wrong_fill_bytes_counts_order_and_failures() {
        let (c, cp) = fixture(qualification_arms().0);
        let (t, tp) = fixture(qualification_arms().1);
        let corruptions: &[fn(&mut Snapshot)] = &[
            |s| s.physical_slot_zero_fill_bytes = 1,
            |s| s.physical_slot_epoch_write_bytes -= 1,
            |s| s.physical_slot_payload_copy_bytes -= 1,
            |s| s.physical_slot_bytes_staged -= 1,
            |s| s.physical_bytes_staged -= 1,
            |s| s.physical_install_completions -= 1,
            |s| s.direct_staging_writes = 0,
            |s| s.full_slot_vec_materializations = 1,
            |s| s.physical_install_attempts += 1,
            |s| s.reservation_attempts += 1,
            |s| s.reservation_successes -= 1,
            |s| s.reservation_failures = 1,
            |s| s.physical_stage_attempts += 1,
            |s| s.physical_stage_completions -= 1,
            |s| s.physical_stage_failures = 1,
            |s| s.direct_staging_failures = 1,
            |s| s.ordered_commit_attempts += 1,
            |s| s.ordered_commit_completions -= 1,
            |s| s.ordered_commit_failures = 1,
            |s| s.ordered_commit_violations = 1,
            |s| s.mapping_publications -= 1,
            |s| s.mapping_unpublications += 1,
            |s| s.unpublished_physical_writes_after_failure = 1,
            |s| s.active_physical_staging = 1,
            |s| s.evidence_accounting_errors = 1,
            |s| s.timing_accounting_errors = 1,
            |s| s.physical_install_total_us += 1,
            |s| s.physical_slot_prepare_us = 1_000,
            |s| s.mapping_publication_us = 1_000,
            |s| s.overlapping_demand_sets = 1,
            |s| s.physical_install_sets += 1,
            |s| s.install_set_width_min = 1,
            |s| s.install_set_width_max = 8,
            |s| s.install_set_width_mean = f64::NAN,
            |s| s.parallel_staging_sets = 0,
            |s| s.parallel_eligible_sets += 1,
            |s| s.parallel_staging_experts -= 1,
            |s| s.parallel_eligible_experts += 1,
            |s| s.singleton_staging_sets = 1,
            |s| s.max_in_flight_physical_staging = 1,
            |s| s.rayon_num_threads = 1,
            |s| s.caller_was_already_rayon_worker = true,
            |s| s.ordered_install_set_behavior_sha256.push('x'),
            |s| s.physical_victim_ids_sha256.push('x'),
            |s| s.physical_residency_identity_sha256.push('x'),
            |s| s.reservation_identity_sha256.push('x'),
            |s| s.selected_route_ids_sha256.push('x'),
            |s| s.physical_missing_ids_sha256.push('x'),
            |s| s.demand_source_request_ids_sha256.push('x'),
            |s| s.demand_source_requests += 1,
            |s| s.source_nvme_reads += 1,
            |s| s.source_nvme_bytes += 1,
        ];
        for (index, corrupt) in corruptions.iter().enumerate() {
            let mut bad = t.clone();
            corrupt(&mut bad);
            assert!(
                !pair_mechanism_gate(&c, &bad, &cp, &tp).passed,
                "corruption {index}"
            );
        }
        for bytes in [0, c.physical_slot_bytes_staged - 1] {
            let mut bad_control = c.clone();
            bad_control.physical_slot_zero_fill_bytes = bytes;
            assert!(!pair_mechanism_gate(&bad_control, &t, &cp, &tp).passed);
        }
        let mut sequential_control = c.clone();
        sequential_control.arm = Arm::Control;
        assert!(!pair_mechanism_gate(&sequential_control, &t, &cp, &tp).passed);
        let mut bad_production = tp;
        bad_production.physical_install_failures = 1;
        assert!(!pair_mechanism_gate(&c, &t, &cp, &bad_production).passed);
        bad_production = tp;
        bad_production.direct_staging_allocation_fallbacks = 1;
        assert!(!pair_mechanism_gate(&c, &t, &cp, &bad_production).passed);
    }

    #[test]
    fn zero_fill_production_rejects_absent_work_and_byte_overflow() {
        let c = crate::engine::empty_physical_zero_fill_test_snapshot(qualification_arms().0);
        let t = crate::engine::empty_physical_zero_fill_test_snapshot(qualification_arms().1);
        let p = GpuNativeProductionPhysicalInstallSnapshot::default();
        assert!(!pair_mechanism_gate(&c, &t, &p, &p).passed);
        let (mut huge, _) = fixture(qualification_arms().1);
        huge.physical_install_completions = u64::MAX;
        assert!(!bytes_exact(&huge));
    }
}
