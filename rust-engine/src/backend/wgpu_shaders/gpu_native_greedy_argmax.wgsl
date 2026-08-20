// GPU-native greedy argmax sampler for one token.
//
// One 64-lane workgroup strides over the vocabulary logits. Per-lane candidates
// are reduced deterministically in shared workgroup memory with an exact
// lower-token-id tie-break. Any non-finite logit latches the LM-head numerical
// failure status bit and writes u32::MAX.

struct PushConstants {
    vocab_size: u32,
    _reserved_0: u32,
    _reserved_1: u32,
    _reserved_2: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> LOGITS: array<f32>;
@group(0) @binding(1) var<storage, read_write> SAMPLED_TOKEN: array<u32>;
struct Status {
    bits: atomic<u32>,
};
@group(0) @binding(2) var<storage, read_write> STATUS: Status;

const WG: u32 = 64u;
const MAX_FINITE_F32: f32 = 3.402823e+38;
const FINITE_NEGATIVE_SENTINEL: f32 = -3.402823e+38;
const INVALID_TOKEN_SENTINEL: u32 = 0xffffffffu;
const GPU_NATIVE_STATUS_LM_HEAD_NUMERICAL_FAILURE: u32 = 16u;

var<workgroup> candidate_vals: array<f32, 64u>;
var<workgroup> candidate_idxs: array<u32, 64u>;
var<workgroup> numerical_failure: atomic<u32>;

fn is_finite(value: f32) -> bool {
    return value == value && abs(value) <= MAX_FINITE_F32;
}

fn latch_numerical_failure() {
    atomicOr(&STATUS.bits, GPU_NATIVE_STATUS_LM_HEAD_NUMERICAL_FAILURE);
    atomicOr(&numerical_failure, GPU_NATIVE_STATUS_LM_HEAD_NUMERICAL_FAILURE);
}

@compute @workgroup_size(64, 1, 1)
fn greedy_argmax_main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    if (lane == 0u) {
        atomicStore(&numerical_failure, 0u);
        SAMPLED_TOKEN[0] = INVALID_TOKEN_SENTINEL;
    }
    workgroupBarrier();

    let prior_status = atomicLoad(&STATUS.bits);
    if (prior_status != 0u) {
        return;
    }

    var best_val = FINITE_NEGATIVE_SENTINEL;
    var best_idx = INVALID_TOKEN_SENTINEL;

    for (var idx = lane; idx < pc.vocab_size; idx = idx + WG) {
        let val = LOGITS[idx];
        if (!is_finite(val)) {
            latch_numerical_failure();
        } else if (best_idx == INVALID_TOKEN_SENTINEL || val > best_val || (val == best_val && idx < best_idx)) {
            best_val = val;
            best_idx = idx;
        }
    }

    candidate_vals[lane] = best_val;
    candidate_idxs[lane] = best_idx;
    workgroupBarrier();

    for (var offset = WG / 2u; offset > 0u; offset = offset / 2u) {
        if (lane < offset) {
            let other_val = candidate_vals[lane + offset];
            let other_idx = candidate_idxs[lane + offset];
            let cur_val = candidate_vals[lane];
            let cur_idx = candidate_idxs[lane];

            if (cur_idx == INVALID_TOKEN_SENTINEL) {
                candidate_vals[lane] = other_val;
                candidate_idxs[lane] = other_idx;
            } else if (other_idx != INVALID_TOKEN_SENTINEL) {
                if (other_val > cur_val || (other_val == cur_val && other_idx < cur_idx)) {
                    candidate_vals[lane] = other_val;
                    candidate_idxs[lane] = other_idx;
                }
            }
        }
        workgroupBarrier();
    }

    if (lane == 0u) {
        let failure = atomicLoad(&numerical_failure);
        let final_status = atomicLoad(&STATUS.bits);
        if (failure != 0u || final_status != 0u || candidate_idxs[0] == INVALID_TOKEN_SENTINEL) {
            SAMPLED_TOKEN[0] = INVALID_TOKEN_SENTINEL;
        } else {
            SAMPLED_TOKEN[0] = candidate_idxs[0];
        }
    }
}
