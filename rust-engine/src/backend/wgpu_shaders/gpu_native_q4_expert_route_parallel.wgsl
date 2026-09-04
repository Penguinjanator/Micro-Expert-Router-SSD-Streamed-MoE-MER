
// PR2-C qualification only. The serial entrypoints above remain unchanged.
fn resolved_route_parallel(route_slot: u32) -> ExpertRoute {
    if (route_slot >= pc.top_k || atomicLoad(&STATUS.bits) != 0u) {
        return ExpertRoute(UNMAPPED, 0u);
    }
    return RESOLVED_LOCATIONS[route_slot];
}

@compute @workgroup_size(64, 1, 1)
fn q4_expert_gate_up_route_parallel_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let route_slot = gid.y;
    if (row >= pc.d_ff || route_slot >= pc.top_k) {
        return;
    }
    let route = resolved_route_parallel(route_slot);
    let location = route.location;
    if (location == UNMAPPED) {
        OUTPUT[route_slot * pc.d_ff + row] = 0.0;
        return;
    }
    let bank = location >> LOCATION_BANK_SHIFT;
    let slot = location & LOCATION_SLOT_MASK;
    let slot_base = slot * pc.slot_stride_bytes;
    let current_slot_epoch = read_bank_word(bank, slot_base >> 2u);
    if (current_slot_epoch == 0u || current_slot_epoch != route.slot_epoch) {
        atomicOr(&STATUS.bits, EXPERT_RESIDENCY_MISS);
        OUTPUT[route_slot * pc.d_ff + row] = 0.0;
        return;
    }
    let payload_base = slot_base + 4u;
    let blocks_per_row = pc.d_model / BLOCK_ELEMS;
    let row_block = row * blocks_per_row;
    let gate = q4_dot(bank, payload_base, row_block, blocks_per_row, 0u);
    let up = q4_dot(
        bank,
        payload_base,
        pc.blocks_per_projection + row_block,
        blocks_per_row,
        0u,
    );
    if (!is_finite(gate) || !is_finite(up)) {
        atomicOr(&STATUS.bits, EXPERT_NUMERICAL_FAILURE);
        OUTPUT[route_slot * pc.d_ff + row] = 0.0;
        return;
    }
    let clamped_gate = clamp(gate, -pc.swiglu_limit, pc.swiglu_limit);
    let activation = (clamped_gate / (1.0 + exp(-clamped_gate))) * up;
    if (!is_finite(activation)) {
        atomicOr(&STATUS.bits, EXPERT_NUMERICAL_FAILURE);
        OUTPUT[route_slot * pc.d_ff + row] = 0.0;
        return;
    }
    OUTPUT[route_slot * pc.d_ff + row] = activation;
}

@compute @workgroup_size(64, 1, 1)
fn q4_expert_down_route_parallel_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let route_slot = gid.y;
    if (row >= pc.d_model || route_slot >= pc.top_k) {
        return;
    }
    let route_output = route_slot * pc.d_model + row;
    let route = resolved_route_parallel(route_slot);
    let location = route.location;
    if (location == UNMAPPED) {
        OUTPUT[route_output] = 0.0;
        return;
    }
    let bank = location >> LOCATION_BANK_SHIFT;
    let slot = location & LOCATION_SLOT_MASK;
    let slot_base = slot * pc.slot_stride_bytes;
    let current_slot_epoch = read_bank_word(bank, slot_base >> 2u);
    if (current_slot_epoch == 0u || current_slot_epoch != route.slot_epoch) {
        atomicOr(&STATUS.bits, EXPERT_RESIDENCY_MISS);
        OUTPUT[route_output] = 0.0;
        return;
    }
    let payload_base = slot_base + 4u;
    let blocks_per_row = pc.d_ff / BLOCK_ELEMS;
    let first_block = 2u * pc.blocks_per_projection + row * blocks_per_row;
    let value = q4_dot(bank, payload_base, first_block, blocks_per_row, route_slot * pc.d_ff);
    if (!is_finite(value)) {
        atomicOr(&STATUS.bits, EXPERT_NUMERICAL_FAILURE);
        OUTPUT[route_output] = 0.0;
        return;
    }
    OUTPUT[route_output] = value;
}
