// GPU-native request-local F32 KV append.
//
// The bound destination buffers belong to one checked layer and are laid out
// as `[absolute_position, width]`.

struct PushConstants {
    width: u32,
    position: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> CURRENT_K: array<f32>;
@group(0) @binding(1) var<storage, read> CURRENT_V: array<f32>;
@group(0) @binding(2) var<storage, read_write> KEY_CACHE: array<f32>;
@group(0) @binding(3) var<storage, read_write> VALUE_CACHE: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn kv_append_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let channel = gid.x;
    if (channel >= pc.width) {
        return;
    }
    let destination = pc.position * pc.width + channel;
    KEY_CACHE[destination] = CURRENT_K[channel];
    VALUE_CACHE[destination] = CURRENT_V[channel];
}
