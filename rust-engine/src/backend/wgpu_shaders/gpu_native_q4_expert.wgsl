// GPU-native Q4_0 expert execution over a four-bank physical slot arena.
//
// Router-selected logical ids never reach the host. A preceding GPU route-
// resolution pass converts them to packed `(bank, slot)` locations. Each
// route is then executed serially so the request-local activation allocation
// can be reused without allocating `top_k * d_ff` elements.

struct PushConstants {
    d_model: u32,
    d_ff: u32,
    blocks_per_projection: u32,
    slot_stride_bytes: u32,
    route_slot: u32,
    top_k: u32,
    swiglu_limit: f32,
    _reserved: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> W0: array<u32>;
@group(0) @binding(1) var<storage, read> W1: array<u32>;
@group(0) @binding(2) var<storage, read> W2: array<u32>;
@group(0) @binding(3) var<storage, read> W3: array<u32>;
@group(0) @binding(4) var<storage, read> INPUT: array<f32>;
@group(0) @binding(5) var<storage, read> RESOLVED_LOCATIONS: array<u32>;
@group(0) @binding(6) var<storage, read_write> OUTPUT: array<f32>;
struct Status {
    bits: atomic<u32>,
};
@group(0) @binding(7) var<storage, read_write> STATUS: Status;

const BLOCK_BYTES: u32 = 18u;
const BLOCK_ELEMS: u32 = 32u;
const LOCATION_BANK_SHIFT: u32 = 30u;
const LOCATION_SLOT_MASK: u32 = 0x3fffffffu;
const UNMAPPED: u32 = 0xffffffffu;
const EXPERT_NUMERICAL_FAILURE: u32 = 8u;
const MAX_FINITE_F32: f32 = 3.402823e+38;

fn is_finite(value: f32) -> bool {
    return value == value && abs(value) <= MAX_FINITE_F32;
}

fn read_bank_byte(bank: u32, byte_offset: u32) -> u32 {
    let word = byte_offset >> 2u;
    let shift = (byte_offset & 3u) * 8u;
    switch bank {
        case 0u: { return (W0[word] >> shift) & 0xffu; }
        case 1u: { return (W1[word] >> shift) & 0xffu; }
        case 2u: { return (W2[word] >> shift) & 0xffu; }
        default: { return (W3[word] >> shift) & 0xffu; }
    }
}

fn q4_dot(
    bank: u32,
    slot_base: u32,
    first_block: u32,
    blocks_per_row: u32,
    input_base: u32,
) -> f32 {
    var sum = 0.0;
    for (var block = 0u; block < blocks_per_row; block += 1u) {
        let byte_offset = slot_base + (first_block + block) * BLOCK_BYTES;
        let scale_bits = read_bank_byte(bank, byte_offset)
            | (read_bank_byte(bank, byte_offset + 1u) << 8u);
        let scale = unpack2x16float(scale_bits).x;
        var partial = 0.0;
        let x_base = input_base + block * BLOCK_ELEMS;
        for (var nibble = 0u; nibble < 16u; nibble += 1u) {
            let packed = read_bank_byte(bank, byte_offset + 2u + nibble);
            partial += (f32(packed & 0xfu) - 8.0) * INPUT[x_base + nibble];
            partial += (f32(packed >> 4u) - 8.0) * INPUT[x_base + nibble + 16u];
        }
        sum += scale * partial;
    }
    return sum;
}

fn resolved_route() -> u32 {
    if (pc.route_slot >= pc.top_k || atomicLoad(&STATUS.bits) != 0u) {
        return UNMAPPED;
    }
    return RESOLVED_LOCATIONS[pc.route_slot];
}

@compute @workgroup_size(64, 1, 1)
fn q4_expert_gate_up_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= pc.d_ff) {
        return;
    }
    let location = resolved_route();
    if (location == UNMAPPED) {
        OUTPUT[row] = 0.0;
        return;
    }
    let bank = location >> LOCATION_BANK_SHIFT;
    let slot = location & LOCATION_SLOT_MASK;
    let slot_base = slot * pc.slot_stride_bytes;
    let blocks_per_row = pc.d_model / BLOCK_ELEMS;
    let row_block = row * blocks_per_row;
    let gate = q4_dot(bank, slot_base, row_block, blocks_per_row, 0u);
    let up = q4_dot(
        bank,
        slot_base,
        pc.blocks_per_projection + row_block,
        blocks_per_row,
        0u,
    );
    if (!is_finite(gate) || !is_finite(up)) {
        atomicOr(&STATUS.bits, EXPERT_NUMERICAL_FAILURE);
        OUTPUT[row] = 0.0;
        return;
    }
    let clamped_gate = clamp(gate, -pc.swiglu_limit, pc.swiglu_limit);
    let activation = (clamped_gate / (1.0 + exp(-clamped_gate))) * up;
    if (!is_finite(activation)) {
        atomicOr(&STATUS.bits, EXPERT_NUMERICAL_FAILURE);
        OUTPUT[row] = 0.0;
        return;
    }
    OUTPUT[row] = activation;
}

@compute @workgroup_size(64, 1, 1)
fn q4_expert_down_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= pc.d_model) {
        return;
    }
    let route_output = pc.route_slot * pc.d_model + row;
    let location = resolved_route();
    if (location == UNMAPPED) {
        OUTPUT[route_output] = 0.0;
        return;
    }
    let bank = location >> LOCATION_BANK_SHIFT;
    let slot = location & LOCATION_SLOT_MASK;
    let slot_base = slot * pc.slot_stride_bytes;
    let blocks_per_row = pc.d_ff / BLOCK_ELEMS;
    let first_block = 2u * pc.blocks_per_projection + row * blocks_per_row;
    let value = q4_dot(bank, slot_base, first_block, blocks_per_row, 0u);
    if (!is_finite(value)) {
        atomicOr(&STATUS.bits, EXPERT_NUMERICAL_FAILURE);
        OUTPUT[route_output] = 0.0;
        return;
    }
    OUTPUT[route_output] = value;
}
