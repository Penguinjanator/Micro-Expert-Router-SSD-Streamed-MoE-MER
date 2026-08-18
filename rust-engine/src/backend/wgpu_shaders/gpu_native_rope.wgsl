// GPU-native per-head rotary positional embedding.
//
// Each invocation rotates one head-relative pair `(i, i + rope_dim / 2)`.
// Channels `[rope_dim, head_dim)` are never addressed and remain unchanged.

struct PushConstants {
    groups: u32,
    head_dim: u32,
    rope_dim: u32,
    position: u32,
    attention_factor_bits: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> INV_FREQ: array<f32>;
@group(0) @binding(1) var<storage, read_write> TARGET: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn rope_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pairs = pc.rope_dim / 2u;
    let flat_pair = gid.x;
    if (flat_pair >= pc.groups * pairs) {
        return;
    }

    let group = flat_pair / pairs;
    let pair = flat_pair % pairs;
    let first = group * pc.head_dim + pair;
    let second = first + pairs;
    let theta = f32(pc.position) * INV_FREQ[pair];
    let scale = bitcast<f32>(pc.attention_factor_bits);
    let sin_theta = sin(theta) * scale;
    let cos_theta = cos(theta) * scale;
    let a = TARGET[first];
    let b = TARGET[second];
    TARGET[first] = a * cos_theta - b * sin_theta;
    TARGET[second] = a * sin_theta + b * cos_theta;
}
