// Copy/dequantize one row of a persistent [vocab_size, d_model] embedding
// directly into GPU-native hidden state. Token ID is a push-constant control
// scalar; the weight and destination never cross the host boundary here.

struct PushConstants {
    local_row: u32,
    global_row: u32,
    cols: u32,
    q8_first_block: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read_write> HIDDEN: array<f32>;

const Q8_0_BLOCK_BYTES: u32 = 34u;
const Q8_0_BLOCK_ELEMS: u32 = 32u;

fn read_weight_byte(byte_offset: u32) -> u32 {
    return (W[byte_offset >> 2u] >> ((byte_offset & 3u) * 8u)) & 0xffu;
}

fn q8_0_value(flat_index: u32) -> f32 {
    let block = flat_index / Q8_0_BLOCK_ELEMS - pc.q8_first_block;
    let in_block = flat_index % Q8_0_BLOCK_ELEMS;
    let block_offset = block * Q8_0_BLOCK_BYTES;
    let scale_bits = read_weight_byte(block_offset)
        | (read_weight_byte(block_offset + 1u) << 8u);
    let scale = unpack2x16float(scale_bits).x;
    let quantized = read_weight_byte(block_offset + 2u + in_block);
    let signed_value = select(i32(quantized), i32(quantized) - 256, quantized >= 128u);
    return scale * f32(signed_value);
}

@compute @workgroup_size(64, 1, 1)
fn f32_embedding_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= pc.cols) {
        return;
    }
    HIDDEN[col] = bitcast<f32>(W[pc.local_row * pc.cols + col]);
}

@compute @workgroup_size(64, 1, 1)
fn q8_0_embedding_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= pc.cols) {
        return;
    }
    HIDDEN[col] = q8_0_value(pc.global_row * pc.cols + col);
}
