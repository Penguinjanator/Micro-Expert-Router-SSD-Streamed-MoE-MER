// Bounded GPU-native Qwen/Mixtral router for one token.
//
// One 64-lane workgroup owns the complete router row. Each lane handles at
// most two of the bounded 128 experts. Softmax reductions are parallel and
// lane 0 performs the tiny deterministic top-8 selection serially.

struct PushConstants {
    num_experts: u32,
    top_k: u32,
    _reserved_0: u32,
    _reserved_1: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> LOGITS: array<f32>;
@group(0) @binding(1) var<storage, read_write> SELECTED_IDS: array<u32>;
@group(0) @binding(2) var<storage, read_write> SELECTED_WEIGHTS: array<f32>;
struct Status {
    bits: atomic<u32>,
};
@group(0) @binding(3) var<storage, read_write> STATUS: Status;

const WG: u32 = 64u;
const MAX_EXPERTS: u32 = 128u;
const MAX_TOP_K: u32 = 8u;
const MAX_FINITE_F32: f32 = 3.402823e+38;
const FINITE_NEGATIVE_SENTINEL: f32 = -3.402823e+38;
const GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE: u32 = 2u;

var<workgroup> scores: array<f32, 128u>;
var<workgroup> reduction: array<f32, 64u>;
var<workgroup> numerical_failure: atomic<u32>;

fn is_finite(value: f32) -> bool {
    return value == value && abs(value) <= MAX_FINITE_F32;
}

fn latch_numerical_failure() {
    atomicOr(&STATUS.bits, GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE);
    atomicOr(&numerical_failure, GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE);
}

fn sanitize_outputs() {
    for (var slot = 0u; slot < pc.top_k; slot = slot + 1u) {
        SELECTED_IDS[slot] = 0u;
        SELECTED_WEIGHTS[slot] = 0.0;
    }
}

@compute @workgroup_size(64, 1, 1)
fn router_topk_main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    if (lane == 0u) {
        atomicStore(&numerical_failure, 0u);
        sanitize_outputs();
    }
    workgroupBarrier();

    var local_max = FINITE_NEGATIVE_SENTINEL;
    for (var expert = lane; expert < pc.num_experts; expert = expert + WG) {
        let logit = LOGITS[expert];
        if (!is_finite(logit)) {
            latch_numerical_failure();
            scores[expert] = 0.0;
        } else {
            scores[expert] = logit;
            local_max = max(local_max, logit);
        }
    }
    reduction[lane] = local_max;
    workgroupBarrier();

    var stride = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            reduction[lane] = max(reduction[lane], reduction[lane + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    let max_logit = reduction[0];
    if (lane == 0u && !is_finite(max_logit)) {
        latch_numerical_failure();
    }
    workgroupBarrier();

    var local_sum = 0.0;
    for (var expert = lane; expert < pc.num_experts; expert = expert + WG) {
        let logit = scores[expert];
        var exponent = 0.0;
        if (is_finite(logit) && is_finite(max_logit)) {
            exponent = exp(logit - max_logit);
            if (!is_finite(exponent) || exponent < 0.0) {
                latch_numerical_failure();
                exponent = 0.0;
            }
        }
        scores[expert] = exponent;
        local_sum = local_sum + exponent;
    }
    if (!is_finite(local_sum)) {
        latch_numerical_failure();
        local_sum = 0.0;
    }
    reduction[lane] = local_sum;
    workgroupBarrier();

    stride = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            reduction[lane] = reduction[lane] + reduction[lane + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    let denominator = reduction[0];
    if (lane == 0u && (!is_finite(denominator) || denominator <= 0.0)) {
        latch_numerical_failure();
    }
    workgroupBarrier();

    if (atomicLoad(&numerical_failure) == 0u) {
        for (var expert = lane; expert < pc.num_experts; expert = expert + WG) {
            let score = scores[expert] / denominator;
            if (!is_finite(score) || score < 0.0) {
                latch_numerical_failure();
                scores[expert] = 0.0;
            } else {
                scores[expert] = score;
            }
        }
    }
    workgroupBarrier();

    if (lane != 0u) {
        return;
    }
    if (atomicLoad(&numerical_failure) != 0u) {
        sanitize_outputs();
        return;
    }

    var selected_sum = 0.0;
    for (var slot = 0u; slot < pc.top_k; slot = slot + 1u) {
        var found = false;
        var best_id = 0u;
        var best_score = 0.0;
        for (var expert = 0u; expert < pc.num_experts; expert = expert + 1u) {
            var already_selected = false;
            for (var previous = 0u; previous < slot; previous = previous + 1u) {
                if (SELECTED_IDS[previous] == expert) {
                    already_selected = true;
                }
            }
            let candidate = scores[expert];
            if (!already_selected
                && (!found
                    || candidate > best_score
                    || (candidate == best_score && expert < best_id))) {
                found = true;
                best_id = expert;
                best_score = candidate;
            }
        }
        if (!found || !is_finite(best_score) || best_score < 0.0) {
            latch_numerical_failure();
        } else {
            SELECTED_IDS[slot] = best_id;
            SELECTED_WEIGHTS[slot] = best_score;
            selected_sum = selected_sum + best_score;
        }
    }

    if (atomicLoad(&numerical_failure) != 0u
        || !is_finite(selected_sum)
        || selected_sum <= 0.0) {
        latch_numerical_failure();
        sanitize_outputs();
        return;
    }

    for (var slot = 0u; slot < pc.top_k; slot = slot + 1u) {
        let selected_weight = SELECTED_WEIGHTS[slot] / selected_sum;
        if (!is_finite(selected_weight) || selected_weight < 0.0) {
            latch_numerical_failure();
        } else {
            SELECTED_WEIGHTS[slot] = selected_weight;
        }
    }
    if (atomicLoad(&numerical_failure) != 0u) {
        sanitize_outputs();
    }
}
