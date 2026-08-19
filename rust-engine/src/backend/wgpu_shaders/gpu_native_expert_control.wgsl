// GPU-only logical-to-physical route resolution and fail-closed combination.

struct ControlPushConstants {
    num_experts: u32,
    top_k: u32,
    active_banks: u32,
    _reserved: u32,
    bank_0_slots: u32,
    bank_1_slots: u32,
    bank_2_slots: u32,
    bank_3_slots: u32,
};
var<push_constant> pc: ControlPushConstants;

@group(0) @binding(0) var<storage, read> SELECTED_IDS: array<u32>;
@group(0) @binding(1) var<storage, read> MAPPING: array<u32>;
@group(0) @binding(2) var<storage, read_write> RESOLVED_LOCATIONS: array<u32>;
struct ResolveStatus {
    bits: atomic<u32>,
};
@group(0) @binding(3) var<storage, read_write> RESOLVE_STATUS: ResolveStatus;

const LOCATION_BANK_SHIFT: u32 = 30u;
const LOCATION_SLOT_MASK: u32 = 0x3fffffffu;
const UNMAPPED: u32 = 0xffffffffu;
const EXPERT_RESIDENCY_MISS: u32 = 4u;

fn bank_capacity(bank: u32) -> u32 {
    switch bank {
        case 0u: { return pc.bank_0_slots; }
        case 1u: { return pc.bank_1_slots; }
        case 2u: { return pc.bank_2_slots; }
        default: { return pc.bank_3_slots; }
    }
}

@compute @workgroup_size(64, 1, 1)
fn expert_route_resolve_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let route_slot = gid.x;
    if (route_slot >= pc.top_k) {
        return;
    }
    RESOLVED_LOCATIONS[route_slot] = UNMAPPED;

    // A previous attention/router failure owns containment. In particular,
    // Slice 6's sanitized logical id 0 must never become a real expert here.
    if (atomicLoad(&RESOLVE_STATUS.bits) != 0u) {
        return;
    }
    let logical_id = SELECTED_IDS[route_slot];
    if (logical_id >= pc.num_experts) {
        atomicOr(&RESOLVE_STATUS.bits, EXPERT_RESIDENCY_MISS);
        return;
    }
    let location = MAPPING[logical_id];
    if (location == UNMAPPED) {
        atomicOr(&RESOLVE_STATUS.bits, EXPERT_RESIDENCY_MISS);
        return;
    }
    let bank = location >> LOCATION_BANK_SHIFT;
    let slot = location & LOCATION_SLOT_MASK;
    if (bank >= pc.active_banks || slot >= bank_capacity(bank)) {
        atomicOr(&RESOLVE_STATUS.bits, EXPERT_RESIDENCY_MISS);
        return;
    }
    RESOLVED_LOCATIONS[route_slot] = location;
}

@group(1) @binding(0) var<storage, read> SELECTED_WEIGHTS: array<f32>;
@group(1) @binding(1) var<storage, read> COMBINE_LOCATIONS: array<u32>;
@group(1) @binding(2) var<storage, read> ROUTE_OUTPUTS: array<f32>;
@group(1) @binding(3) var<storage, read_write> COMBINED: array<f32>;
struct CombineStatus {
    bits: atomic<u32>,
};
@group(1) @binding(4) var<storage, read_write> COMBINE_STATUS: CombineStatus;

const EXPERT_NUMERICAL_FAILURE: u32 = 8u;
const MAX_FINITE_F32: f32 = 3.402823e+38;

fn is_finite(value: f32) -> bool {
    return value == value && abs(value) <= MAX_FINITE_F32;
}

// This deliberately bounded scalar validation dispatch completes before the
// parallel combine dispatch, making one bad route fail the complete mixture.
@compute @workgroup_size(1, 1, 1)
fn expert_validate_main() {
    if (atomicLoad(&COMBINE_STATUS.bits) != 0u) {
        return;
    }
    var selected_weight_sum = 0.0;
    for (var route = 0u; route < pc.top_k; route += 1u) {
        if (COMBINE_LOCATIONS[route] == UNMAPPED) {
            atomicOr(&COMBINE_STATUS.bits, EXPERT_RESIDENCY_MISS);
            return;
        }
        let selected_weight = SELECTED_WEIGHTS[route];
        if (!is_finite(selected_weight) || selected_weight < 0.0) {
            atomicOr(&COMBINE_STATUS.bits, EXPERT_NUMERICAL_FAILURE);
            return;
        }
        selected_weight_sum += selected_weight;
        let route_base = route * pc.num_experts;
        for (var element = 0u; element < pc.num_experts; element += 1u) {
            if (!is_finite(ROUTE_OUTPUTS[route_base + element])) {
                atomicOr(&COMBINE_STATUS.bits, EXPERT_NUMERICAL_FAILURE);
                return;
            }
        }
    }
    if (!is_finite(selected_weight_sum) || selected_weight_sum <= 0.0) {
        atomicOr(&COMBINE_STATUS.bits, EXPERT_NUMERICAL_FAILURE);
    }
}

@compute @workgroup_size(64, 1, 1)
fn expert_combine_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let element = gid.x;
    if (element >= pc.num_experts) {
        return;
    }
    if (atomicLoad(&COMBINE_STATUS.bits) != 0u) {
        COMBINED[element] = 0.0;
        return;
    }
    var value = 0.0;
    for (var route = 0u; route < pc.top_k; route += 1u) {
        value += SELECTED_WEIGHTS[route]
            * ROUTE_OUTPUTS[route * pc.num_experts + element];
    }
    if (!is_finite(value)) {
        atomicOr(&COMBINE_STATUS.bits, EXPERT_NUMERICAL_FAILURE);
        COMBINED[element] = 0.0;
        return;
    }
    COMBINED[element] = value;
}

// A distinct dispatch is required after the parallel combine: if one lane
// latches a numerical failure, every other lane's already-written value must
// still be erased before residual completion.
@compute @workgroup_size(64, 1, 1)
fn expert_contain_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < pc.num_experts && atomicLoad(&COMBINE_STATUS.bits) != 0u) {
        COMBINED[gid.x] = 0.0;
    }
}
