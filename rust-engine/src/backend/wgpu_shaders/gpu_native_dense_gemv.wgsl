// GPU-native row-major GEMV over persistent F32 or repository Q8_0 weights.
//
// Matrix convention: W is [rows, cols], X is [cols], OUT is [rows]. Q8_0
// is one flat tensor stream, so a matrix row may begin/end inside a block.
// Each 34-byte block stores an f16-LE scale followed by 32 signed i8 values.

struct PushConstants {
    rows: u32,
    cols: u32,
    global_row_base: u32,
    q8_first_block: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;

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
fn f32_gemv_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_row = gid.x;
    if (local_row >= pc.rows) {
        return;
    }
    let row_start = local_row * pc.cols;
    var sum = 0.0;
    for (var col = 0u; col < pc.cols; col = col + 1u) {
        sum = sum + bitcast<f32>(W[row_start + col]) * X[col];
    }
    OUT[pc.global_row_base + local_row] = sum;
}

@compute @workgroup_size(64, 1, 1)
fn q8_0_gemv_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_row = gid.x;
    if (local_row >= pc.rows) {
        return;
    }
    let global_row = pc.global_row_base + local_row;
    let row_start = global_row * pc.cols;
    var sum = 0.0;
    for (var col = 0u; col < pc.cols; col = col + 1u) {
        sum = sum + q8_0_value(row_start + col) * X[col];
    }
    OUT[global_row] = sum;
}
