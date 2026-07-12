//! AVX2 + FMA kernels — feature-less auto-escalation path.
//!
//! Compiled unconditionally on `x86_64` (no cargo feature gate) so a
//! single binary deployed across a heterogeneous fleet automatically
//! benefits from AVX2 on any host that supports it. Every entry point
//! is `unsafe` because it relies on `#[target_feature(enable =
//! "avx2,fma")]`; callers gate dispatch on the runtime probe in
//! [`super::detect`] so these routines never execute on a CPU that
//! doesn't support them.
//!
//! Results are bit-equivalent to the [`super::scalar`] reference up
//! to floating-point reduction reordering (about 1 ULP per ~8-wide
//! accumulator, well under the engine's `1e-3` tolerance).

#![cfg(target_arch = "x86_64")]

use super::{q8_0_scale_from_ptr, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};
use std::arch::x86_64::*;

/// AVX2 f32 dot product. 8-wide FMA accumulator, scalar tail.
///
/// # Safety
///
/// Caller must guarantee the CPU supports `avx2 + fma`. The
/// dispatcher in [`super::dot_f32`] checks this exactly once at
/// startup via [`super::cpu_features`].
///
/// `a` and `b` are borrowed slices whose backing storage remains
/// valid for the duration of this call (the standard Rust borrow
/// rules); the kernel reads through `_mm256_loadu_ps` (no alignment
/// requirement on the pointer), writes nothing, and uses a separate
/// scalar loop for the < 8 trailing elements, so no out-of-bounds
/// access is possible for any `a.len() == b.len()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    // Defensive bound-check (gist feedback #1.1): the SIMD `while i +
    // 8 <= n` loop is already safe for any `n` (including `n == 0`
    // and `n == 1`, where it iterates zero times and the scalar tail
    // handles every element), but pinning the invariants here makes
    // a future refactor that misuses pointer arithmetic fail loudly.
    debug_assert!(
        n == 0 || (a.as_ptr() as usize % core::mem::align_of::<f32>() == 0
            && b.as_ptr() as usize % core::mem::align_of::<f32>() == 0),
        "dot_f32_avx2: non-empty slices must be `f32`-aligned"
    );
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` guarantees the eight floats from offset
        // `i` are in bounds for both slices (which we just bound-
        // checked against `a.len() == b.len() == n`); `loadu_ps`
        // imposes no alignment requirement on the pointer.
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    // Horizontal sum of the 8-wide accumulator.
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf);
    let mut sum = _mm_cvtss_f32(sums);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(acc: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf))
}

/// Row-wide GGUF Q8_0 × F32 dot product.
///
/// Four independent accumulators span all blocks in the row and are reduced
/// only once. This avoids the previous horizontal reduction for every
/// 32-weight block, which is particularly expensive at Qwen's 2,048/768
/// widths.
///
/// # Safety
/// The CPU must support AVX2+FMA. `x.len()` must be divisible by 32 and
/// `row_blocks.len()` must equal `x.len()/32*34`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn q8_0_row_dot_avx2(row_blocks: &[u8], x: &[f32]) -> f32 {
    debug_assert!(x.len().is_multiple_of(Q8_0_BLOCK_ELEMS));
    debug_assert_eq!(
        row_blocks.len(),
        x.len() / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES
    );
    let mut acc = [_mm256_setzero_ps(); 4];
    for block_index in 0..x.len() / Q8_0_BLOCK_ELEMS {
        let block = row_blocks.as_ptr().add(block_index * Q8_0_BLOCK_BYTES);
        let q = block.add(2);
        let xv = x.as_ptr().add(block_index * Q8_0_BLOCK_ELEMS);
        let scale = _mm256_set1_ps(q8_0_scale_from_ptr(block));
        for lane in 0..4 {
            let offset = lane * 8;
            let packed = _mm_loadl_epi64(q.add(offset).cast::<__m128i>());
            let weights = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed)), scale);
            acc[lane] = _mm256_fmadd_ps(
                weights,
                _mm256_loadu_ps(xv.add(offset)),
                acc[lane],
            );
        }
    }
    let acc = _mm256_add_ps(
        _mm256_add_ps(acc[0], acc[1]),
        _mm256_add_ps(acc[2], acc[3]),
    );
    horizontal_sum_avx2(acc)
}

/// Fused gate/up row traversal for GGUF Q8_0.
///
/// Each 8-wide activation vector is loaded once and feeds independent gate
/// and up accumulators. Both rows are reduced only after their complete
/// traversal.
///
/// # Safety
/// The CPU must support AVX2+FMA. Gate/up ranges must have identical length;
/// `x.len()` must be divisible by 32; each range must be `x.len()/32*34`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn q8_0_gate_up_row_avx2(
    gate_blocks: &[u8],
    up_blocks: &[u8],
    x: &[f32],
) -> (f32, f32) {
    debug_assert_eq!(gate_blocks.len(), up_blocks.len());
    debug_assert!(x.len().is_multiple_of(Q8_0_BLOCK_ELEMS));
    debug_assert_eq!(
        gate_blocks.len(),
        x.len() / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES
    );
    let mut gate_acc = [_mm256_setzero_ps(); 4];
    let mut up_acc = [_mm256_setzero_ps(); 4];
    for block_index in 0..x.len() / Q8_0_BLOCK_ELEMS {
        let gate_block = gate_blocks
            .as_ptr()
            .add(block_index * Q8_0_BLOCK_BYTES);
        let up_block = up_blocks.as_ptr().add(block_index * Q8_0_BLOCK_BYTES);
        let gate_q = gate_block.add(2);
        let up_q = up_block.add(2);
        let xv = x.as_ptr().add(block_index * Q8_0_BLOCK_ELEMS);
        let gate_scale = _mm256_set1_ps(q8_0_scale_from_ptr(gate_block));
        let up_scale = _mm256_set1_ps(q8_0_scale_from_ptr(up_block));
        for lane in 0..4 {
            let offset = lane * 8;
            let activations = _mm256_loadu_ps(xv.add(offset));
            let gate_packed = _mm_loadl_epi64(gate_q.add(offset).cast::<__m128i>());
            let up_packed = _mm_loadl_epi64(up_q.add(offset).cast::<__m128i>());
            let gate_weights = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(gate_packed)),
                gate_scale,
            );
            let up_weights = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(up_packed)),
                up_scale,
            );
            gate_acc[lane] = _mm256_fmadd_ps(gate_weights, activations, gate_acc[lane]);
            up_acc[lane] = _mm256_fmadd_ps(up_weights, activations, up_acc[lane]);
        }
    }
    let gate_acc = _mm256_add_ps(
        _mm256_add_ps(gate_acc[0], gate_acc[1]),
        _mm256_add_ps(gate_acc[2], gate_acc[3]),
    );
    let up_acc = _mm256_add_ps(
        _mm256_add_ps(up_acc[0], up_acc[1]),
        _mm256_add_ps(up_acc[2], up_acc[3]),
    );
    (horizontal_sum_avx2(gate_acc), horizontal_sum_avx2(up_acc))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q8_row(width: usize, case: usize, seed: u64) -> Vec<u8> {
        let blocks = width / Q8_0_BLOCK_ELEMS;
        let mut out = vec![0u8; blocks * Q8_0_BLOCK_BYTES];
        let mut state = seed;
        for block in 0..blocks {
            let start = block * Q8_0_BLOCK_BYTES;
            let scale_bits = match case {
                0 => 0,      // zero f16 scale
                1 => 1,      // smallest positive f16 subnormal
                2 => 0x7bff, // largest finite f16
                6 => 0x0400, // smallest positive normal f16
                _ => half::f16::from_f32(0.003 + block as f32 * 0.00001).to_bits(),
            };
            out[start..start + 2].copy_from_slice(&scale_bits.to_le_bytes());
            for i in 0..Q8_0_BLOCK_ELEMS {
                let q = match case {
                    3 => 0,
                    4 => if (i + block) % 2 == 0 { i8::MIN } else { i8::MAX },
                    5 => if (i + block) % 2 == 0 { -1 } else { 1 },
                    _ => {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        state as i8
                    }
                };
                out[start + 2 + i] = q as u8;
            }
        }
        out
    }

    fn assert_close(name: &str, actual: f32, expected: f32) {
        let tolerance = 2e-4 + expected.abs() * 2e-5;
        assert!(
            (actual - expected).abs() <= tolerance,
            "{name}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }

    fn q8_scalar(row: &[u8], x: &[f32]) -> f32 {
        let mut out = 0.0;
        for (block, activations) in row
            .chunks_exact(Q8_0_BLOCK_BYTES)
            .zip(x.chunks_exact(Q8_0_BLOCK_ELEMS))
        {
            let scale =
                half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            for i in 0..Q8_0_BLOCK_ELEMS {
                out += scale * (block[2 + i] as i8 as f32) * activations[i];
            }
        }
        out
    }

    #[test]
    fn dot_f32_avx2_matches_scalar_when_supported() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let a: Vec<f32> = (0..123).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let b: Vec<f32> = (0..123).map(|i| ((i as f32) * 0.7).cos()).collect();
        // SAFETY: branch guarded above on the CPU feature probe.
        let lhs = unsafe { dot_f32_avx2(&a, &b) };
        let rhs = crate::kernels::scalar::dot_f32(&a, &b);
        assert!((lhs - rhs).abs() <= 1e-3);
    }

    #[test]
    fn q8_row_and_fused_gate_up_avx2_match_scalar_when_supported() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        for width in [32usize, 64, 96, 768, 2048] {
            for case in 0..7 {
                for seed in [1u64, 0x1234_5678, 0xfeed_beef] {
                    let gate = q8_row(width, case, seed);
                    let up = q8_row(width, case, seed ^ 0xa5a5_5a5a);
                    let x: Vec<f32> = (0..width)
                        .map(|i| ((i as f32) * 0.013).sin() * 0.01)
                        .collect();
                    let gate_ref = q8_scalar(&gate, &x);
                    let up_ref = q8_scalar(&up, &x);
                    let gate_dot = unsafe { q8_0_row_dot_avx2(&gate, &x) };
                    let up_dot = unsafe { q8_0_row_dot_avx2(&up, &x) };
                    let (gate_fused, up_fused) =
                        unsafe { q8_0_gate_up_row_avx2(&gate, &up, &x) };
                    let label = format!("width={width} case={case} seed={seed}");
                    assert_close(&format!("gate-dot {label}"), gate_dot, gate_ref);
                    assert_close(&format!("up/down-dot {label}"), up_dot, up_ref);
                    assert_close(&format!("gate-fused {label}"), gate_fused, gate_ref);
                    assert_close(&format!("up-fused {label}"), up_fused, up_ref);
                }
            }
        }
    }
}
