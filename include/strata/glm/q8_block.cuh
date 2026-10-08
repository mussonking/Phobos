#pragma once
// Device helpers for kernels that write glm_matmul's Q8 prepared input themselves (wrapped by matmul_input_q8),
// which saves the quantize launch of matmul_prepare. Layout for C = T * batch columns of n_in elements: int8
// qs [C][n_in], then float d [C][n_in / 32] at byte offset q8_scales_offset(C, n_in) of the same buffer
// (matmul_prepare_bytes(n_in, T, batch) bytes in total). Rounding is ggml's CPU quantize_row_q8_0 (AVX path):
// id = 127 / amax, q = round-half-even(x * id), d = fp16(amax / 127) widened back to float. matmul_prepare uses
// these same helpers, so a fused producer gives bit-identical activations.

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace strata::glm {

__host__ __device__ constexpr size_t q8_scales_offset(size_t C, int n_in) {
    return (C * (size_t) n_in + 255) & ~size_t(255);
}

// Max over groups of G consecutive lanes (G a power of two <= 32, groups aligned to G); all 32 lanes must call it.
template <int G>
__device__ __forceinline__ float q8_group_amax(float a) {
#pragma unroll
    for (int o = G / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    return a;
}

// Per-block constants from the block's amax: id multiplies the values, d is the stored scale.
__device__ __forceinline__ float q8_inv_scale(float amax) { return amax != 0.0f ? __fdiv_rn(127.0f, amax) : 0.0f; }
__device__ __forceinline__ float q8_scale(float amax) { return __half2float(__float2half_rn(__fdiv_rn(amax, 127.0f))); }
__device__ __forceinline__ int8_t q8_quant(float v, float id) { return (int8_t) __float2int_rn(__fmul_rn(v, id)); }

}  // namespace strata::glm
