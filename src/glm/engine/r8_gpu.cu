// In-place R8 -> GGUF conversion of an uploaded expert plane. The host arena keeps the experts in
// pack's R8 layout (rows interleaved by groups of 8, for the CPU miss kernels); the VRAM slots stay GGUF, so every
// upload of R8 bytes runs this on the copy's stream (routed_upload_expert). One CUDA block per 8-row group, through
// shared memory: the group's bytes are exactly the GGUF rows' bytes, nothing outside the group is touched.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "routed.hpp"

namespace strata::glm {
namespace {
constexpr int kXsTile = 1088, kNlTile = 144, kXsBlock = 136, kNlBlock = 18;

// 16 GGUF qs bytes (qs[j] = w[j] | w[j + 16] << 4) of row r from a group's two 64-B halves: lanes 2r, 2r+1 of a half
// are its bytes 8r..8r+7 = B, with w[kk] = B[kk] & 15 (kk < 8) or B[kk - 8] >> 4 (kk >= 8)
__device__ __forceinline__ void qs16(const uint8_t * halves, int r, uint8_t * out) {
    const uint64_t b0 = *reinterpret_cast<const uint64_t *>(halves + 8 * r);
    const uint64_t b1 = *reinterpret_cast<const uint64_t *>(halves + 64 + 8 * r);
    const uint64_t lo = 0x0f0f0f0f0f0f0f0full;
    const uint64_t q0 = (b0 & lo) | ((b1 & lo) << 4), q1 = ((b0 >> 4) & lo) | (b1 & ~lo);
    memcpy(out, &q0, 8);
    memcpy(out + 8, &q1, 8);
}
// load the row group (bytes) into in[], build the GGUF image in out[], store it back (uint4, coalesced)
__device__ __forceinline__ void load_group(const uint8_t * g, uint8_t * in, int bytes) {
    for (int i = threadIdx.x * 16; i < bytes; i += blockDim.x * 16)
        *reinterpret_cast<uint4 *>(in + i) = *reinterpret_cast<const uint4 *>(g + i);
    __syncthreads();
}
__device__ __forceinline__ void store_group(uint8_t * g, const uint8_t * out, int bytes) {
    __syncthreads();
    for (int i = threadIdx.x * 16; i < bytes; i += blockDim.x * 16)
        *reinterpret_cast<uint4 *>(g + i) = *reinterpret_cast<const uint4 *>(out + i);
}

// plane: n_rows (multiple of 8) R8 rows of W; becomes GGUF in place
template <int W>
__global__ void r8_to_gguf_xs(uint8_t * plane) {
    constexpr int nb = W / 256, row = nb * kXsBlock, bytes = 8 * row;
    extern __shared__ __align__(16) uint8_t sm[];
    uint8_t *in = sm, *out = sm + bytes;
    uint8_t * g = plane + size_t(blockIdx.x) * bytes;
    load_group(g, in, bytes);
    for (int u = threadIdx.x; u < 8 * nb * 8; u += blockDim.x) {   // (row r, block b, group gq): 16 qs bytes
        const int r = u / (nb * 8), b = (u / 8) % nb, gq = u % 8;
        qs16(in + b * kXsTile + 64 + 128 * gq, r, out + r * row + b * kXsBlock + 8 + 16 * gq);
    }
    for (int u = threadIdx.x; u < 8 * nb; u += blockDim.x) {   // (row r, block b): d, scales_h, scales_l
        const int r = u / nb, b = u % nb;
        const uint8_t * t = in + b * kXsTile;
        uint8_t * o = out + r * row + b * kXsBlock;
        int S[8];
        for (int gq = 0; gq < 8; ++gq) {
            const int j = 8 * gq + r;
            S[gq] = (j < 32 ? t[16 + j] & 15 : t[16 + j - 32] >> 4) | ((t[48 + (j & 15)] >> (2 * (j >> 4))) & 3) << 4;
        }
        int sh = 0;
        for (int gq = 0; gq < 8; ++gq) sh |= (S[gq] >> 4) << (2 * gq);
        o[0] = t[2 * r];
        o[1] = t[2 * r + 1];
        o[2] = uint8_t(sh);
        o[3] = uint8_t(sh >> 8);
        for (int k = 0; k < 4; ++k) o[4 + k] = uint8_t((S[2 * k] & 15) | (S[2 * k + 1] & 15) << 4);
    }
    store_group(g, out, bytes);
}
template <int W>
__global__ void r8_to_gguf_nl(uint8_t * plane) {
    constexpr int nb = W / 32, row = nb * kNlBlock, bytes = 8 * row;
    extern __shared__ __align__(16) uint8_t sm[];
    uint8_t *in = sm, *out = sm + bytes;
    uint8_t * g = plane + size_t(blockIdx.x) * bytes;
    load_group(g, in, bytes);
    for (int u = threadIdx.x; u < 8 * nb; u += blockDim.x) {   // (row r, block b)
        const int r = u / nb, b = u % nb;
        const uint8_t * t = in + b * kNlTile;
        uint8_t * o = out + r * row + b * kNlBlock;
        o[0] = t[2 * r];
        o[1] = t[2 * r + 1];
        qs16(t + 16, r, o + 2);
    }
    store_group(g, out, bytes);
}
}  // namespace

// GLM-5.3-Flash's widths: gate / up rows of 4096 (IQ4_XS), down rows of 2048 (IQ4_NL); shared memory per block (in +
// out) 34816 / 18432 B, under the 48 KB default
void r8_plane_to_gguf(void * plane, bool xs, int width, int n_rows, cudaStream_t s) {
    auto * p = static_cast<uint8_t *>(plane);
    if (n_rows % 8) throw std::runtime_error("r8_plane_to_gguf: rows not a multiple of 8");
    if (xs && width == 4096) r8_to_gguf_xs<4096><<<n_rows / 8, 256, 2 * 8 * 16 * kXsBlock, s>>>(p);
    else if (!xs && width == 2048) r8_to_gguf_nl<2048><<<n_rows / 8, 256, 2 * 8 * 64 * kNlBlock, s>>>(p);
    else throw std::runtime_error("r8_plane_to_gguf: unsupported width " + std::to_string(width));
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("r8_plane_to_gguf: ") + cudaGetErrorString(e));
}

}  // namespace strata::glm
