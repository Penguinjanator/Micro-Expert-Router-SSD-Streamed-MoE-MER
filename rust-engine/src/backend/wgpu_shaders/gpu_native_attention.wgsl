// GPU-native incremental causal attention over one request-local KV layer.
//
// One 32-lane workgroup owns one query head. Sequence positions are tiled in
// groups of 32 and merged with stable online-softmax state. Each lane owns a
// disjoint set of context channels, so the STORAGE-backed running weighted-V
// accumulator needs no atomics or compile-time head-dimension array.

struct PushConstants {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> KEY_CACHE: array<f32>;
@group(0) @binding(2) var<storage, read> VALUE_CACHE: array<f32>;
@group(0) @binding(3) var<storage, read_write> CONTEXT: array<f32>;
struct Status {
    bits: atomic<u32>,
};
@group(0) @binding(4) var<storage, read_write> STATUS: Status;

const WG: u32 = 32u;
const FINITE_NEGATIVE_SENTINEL: f32 = -3.402823e+38;
const GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE: u32 = 1u;

var<workgroup> running_max: f32;
var<workgroup> running_denominator: f32;
var<workgroup> numerical_failure: atomic<u32>;
var<workgroup> tile_scores: array<f32, 32u>;
var<workgroup> tile_maxima: array<f32, 32u>;
var<workgroup> tile_denominators: array<f32, 32u>;
var<workgroup> tile_weights: array<f32, 32u>;

fn is_finite(value: f32) -> bool {
    return value == value && abs(value) <= 3.402823e+38;
}

fn latch_numerical_failure() {
    atomicOr(&STATUS.bits, GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE);
    atomicOr(&numerical_failure, GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE);
}

@compute @workgroup_size(32, 1, 1)
fn causal_attention_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let query_head = workgroup_id.x;
    let lane = local_id.x;
    if (query_head >= pc.num_heads) {
        return;
    }

    if (lane == 0u) {
        running_max = FINITE_NEGATIVE_SENTINEL;
        running_denominator = 0.0;
        atomicStore(&numerical_failure, 0u);
    }
    let context_base = query_head * pc.head_dim;
    for (var channel = lane; channel < pc.head_dim; channel += WG) {
        CONTEXT[context_base + channel] = 0.0;
    }
    storageBarrier();
    workgroupBarrier();

    let kv_width = pc.num_kv_heads * pc.head_dim;
    // Equivalent to query_head * num_kv_heads / num_heads for the validated
    // divisible GQA geometry, without a potentially overflowing product.
    let queries_per_kv_head = pc.num_heads / pc.num_kv_heads;
    let kv_head = query_head / queries_per_kv_head;
    let query_base = query_head * pc.head_dim;
    let scale = 1.0 / sqrt(f32(pc.head_dim));
    let tile_count = (pc.seq_len - 1u) / WG + 1u;

    for (var tile = 0u; tile < tile_count; tile++) {
        let position = tile * WG + lane;
        let causally_valid = position < pc.seq_len;
        let valid_positions = min(WG, pc.seq_len - tile * WG);

        var score = FINITE_NEGATIVE_SENTINEL;
        var numerically_valid = false;
        if (causally_valid) {
            score = 0.0;
            let key_base = position * kv_width + kv_head * pc.head_dim;
            for (var channel = 0u; channel < pc.head_dim; channel++) {
                score += Q[query_base + channel] * KEY_CACHE[key_base + channel];
            }
            score *= scale;
            numerically_valid = is_finite(score);
            if (!numerically_valid) {
                latch_numerical_failure();
                score = FINITE_NEGATIVE_SENTINEL;
            }
        }
        tile_scores[lane] = score;
        tile_maxima[lane] = score;
        tile_denominators[lane] = select(0.0, 1.0, numerically_valid);
        workgroupBarrier();

        for (var step = 0u; step < 5u; step++) {
            let half = 16u >> step;
            if (lane < half) {
                let first_max = tile_maxima[lane];
                let first_denominator = tile_denominators[lane];
                let second_max = tile_maxima[lane + half];
                let second_denominator = tile_denominators[lane + half];
                let merged_max = max(first_max, second_max);
                tile_maxima[lane] = merged_max;
                tile_denominators[lane] =
                    first_denominator * exp(first_max - merged_max)
                    + second_denominator * exp(second_max - merged_max);
            }
            workgroupBarrier();
        }

        let old_max = running_max;
        let current_tile_max = tile_maxima[0];
        let new_max = max(old_max, current_tile_max);
        let old_factor = exp(old_max - new_max);
        tile_weights[lane] = select(
            0.0,
            exp(tile_scores[lane] - new_max),
            numerically_valid,
        );
        workgroupBarrier();

        for (var channel = lane; channel < pc.head_dim; channel += WG) {
            let context_index = context_base + channel;
            var weighted_value = CONTEXT[context_index] * old_factor;
            for (var local_position = 0u; local_position < valid_positions; local_position++) {
                let absolute_position = tile * WG + local_position;
                let value_index = absolute_position * kv_width
                    + kv_head * pc.head_dim
                    + channel;
                weighted_value += tile_weights[local_position] * VALUE_CACHE[value_index];
            }
            CONTEXT[context_index] = weighted_value;
        }
        storageBarrier();
        workgroupBarrier();

        if (lane == 0u) {
            running_denominator = running_denominator * old_factor
                + tile_denominators[0] * exp(current_tile_max - new_max);
            running_max = new_max;
        }
        workgroupBarrier();
    }

    if (lane == 0u && (!is_finite(running_denominator) || running_denominator <= 0.0)) {
        latch_numerical_failure();
    }
    workgroupBarrier();

    let head_failed = atomicLoad(&numerical_failure) != 0u;
    var inverse_denominator = 0.0;
    if (!head_failed) {
        inverse_denominator = 1.0 / running_denominator;
    }
    for (var channel = lane; channel < pc.head_dim; channel += WG) {
        let context_index = context_base + channel;
        if (head_failed) {
            CONTEXT[context_index] = 0.0;
        } else {
            CONTEXT[context_index] *= inverse_denominator;
        }
    }
}
