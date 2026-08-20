// Encoder-only request-status control for the future GPU-native replay seam.

struct RequestStatus {
    bits: atomic<u32>,
};

@group(0) @binding(0) var<storage, read_write> STATUS: RequestStatus;

const EXPERT_RESIDENCY_RETRYABLE: u32 = 4u;

@compute @workgroup_size(1, 1, 1)
fn clear_retryable_expert_residency_main() {
    atomicAnd(&STATUS.bits, ~EXPERT_RESIDENCY_RETRYABLE);
}
