// GPU-native RMSNorm and residual-state transitions.
//
// RMSNorm launches one workgroup per independently normalised group. Each
// thread accumulates a strided f32 partial sum, then the workgroup reduces the
// 64 partials. The F32 gain vector is shared across groups. The capture entry
// also preserves the old target value in RESIDUAL before overwriting TARGET.
// The power-of-two reduction width is fixed at 64: workgroup size, scratch
// length, input stride, and the initial half-stride of 32 must change together.

struct PushConstants {
    groups: u32,
    group_width: u32,
    epsilon_bits: u32,
    _reserved: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> PARAMETER: array<f32>;
@group(0) @binding(1) var<storage, read_write> TARGET: array<f32>;
@group(0) @binding(2) var<storage, read_write> RESIDUAL: array<f32>;

var<workgroup> SQUARED_SUMS: array<f32, 64>;

fn rms_inverse(group: u32, local_index: u32) -> f32 {
    let group_start = group * pc.group_width;
    var partial_sum = 0.0;
    for (
        var index = local_index;
        index < pc.group_width;
        index = index + 64u
    ) {
        let value = TARGET[group_start + index];
        partial_sum = partial_sum + value * value;
    }
    SQUARED_SUMS[local_index] = partial_sum;
    workgroupBarrier();

    var stride = 32u;
    loop {
        if (local_index < stride) {
            SQUARED_SUMS[local_index] =
                SQUARED_SUMS[local_index] + SQUARED_SUMS[local_index + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride >> 1u;
    }

    let mean_square = SQUARED_SUMS[0] / f32(pc.group_width);
    return 1.0 / sqrt(mean_square + bitcast<f32>(pc.epsilon_bits));
}

fn rms_norm_group_capture(group: u32, local_index: u32) {
    let group_start = group * pc.group_width;
    let inverse_rms = rms_inverse(group, local_index);
    for (
        var index = local_index;
        index < pc.group_width;
        index = index + 64u
    ) {
        let target_index = group_start + index;
        let value = TARGET[target_index];
        RESIDUAL[target_index] = value;
        TARGET[target_index] = value * inverse_rms * PARAMETER[index];
    }
}

fn rms_norm_group_in_place(group: u32, local_index: u32) {
    let group_start = group * pc.group_width;
    let inverse_rms = rms_inverse(group, local_index);
    for (
        var index = local_index;
        index < pc.group_width;
        index = index + 64u
    ) {
        let target_index = group_start + index;
        let value = TARGET[target_index];
        TARGET[target_index] = value * inverse_rms * PARAMETER[index];
    }
}

@compute @workgroup_size(64, 1, 1)
fn rms_norm_capture_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    if (workgroup_id.x >= pc.groups) {
        return;
    }
    rms_norm_group_capture(workgroup_id.x, local_index);
}

@compute @workgroup_size(64, 1, 1)
fn rms_norm_in_place_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    if (workgroup_id.x >= pc.groups) {
        return;
    }
    rms_norm_group_in_place(workgroup_id.x, local_index);
}

@compute @workgroup_size(64, 1, 1)
fn residual_add_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= pc.group_width) {
        return;
    }
    TARGET[index] = RESIDUAL[index] + PARAMETER[index];
}
