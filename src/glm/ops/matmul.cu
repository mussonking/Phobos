// src/glm/ops/matmul.cu - see include/strata/glm/matmul.hpp.
#include "strata/glm/matmul.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "strata/glm/q8_block.cuh"

namespace strata::glm {
namespace {

constexpr int kTypeF32 = 0, kTypeF16 = 1, kTypeQ8_0 = 8, kTypeBF16 = 30;
constexpr int kQ8Block = 34;   // ggml block_q8_0: half d + 32 int8
constexpr int kThreads = 128;

enum Kind { kNone = 0, kQ8 = 1, kF16 = 2 };

[[noreturn]] void fail(const std::string & what) { throw std::runtime_error("glm_matmul: " + what); }

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
}
void ck(cublasStatus_t s, const char * what) {
    if (s != CUBLAS_STATUS_SUCCESS) fail(std::string(what) + ": cuBLAS status " + std::to_string((int) s));
}

size_t align256(size_t n) { return (n + 255) & ~size_t(255); }

bool is_q8(int type) { return type == kTypeQ8_0 || type == kTypeQ8_0R; }
int elem_size(int type) { return type == kTypeF32 ? 4 : (type == kTypeF16 || type == kTypeBF16) ? 2 : 1; }

struct Shape {
    int n_in, n_out, batch;
};
Shape shape_of(const WeightRef & w) {
    if (w.ne[3] != 1) fail("4D weights are not supported");
    Shape s{(int) w.ne[0], (int) w.ne[1], (int) w.ne[2]};
    if (s.n_in <= 0 || s.n_out <= 0 || s.batch <= 0) fail("empty weight");
    if (!is_q8(w.type) && w.type != kTypeF32 && w.type != kTypeF16 && w.type != kTypeBF16)
        fail("unsupported weight type " + std::to_string(w.type));
    if (is_q8(w.type) && s.n_in % 32) fail("Q8_0 n_in must be a multiple of 32");
    return s;
}

// Q8_0R (repacked): int8 quants [rows][n_in], then the fp16 scales [rows][n_in / 32] at align256(rows * n_in).
size_t q8r_scale_offset(int64_t rows, int n_in) { return align256((size_t) rows * n_in); }
size_t q8r_bytes(int64_t rows, int n_in) { return q8r_scale_offset(rows, n_in) + (size_t) rows * (n_in / 32) * 2; }

// ---------------------------------------------------------------------------------------------------------
// Activation preparation. Column c = b * T + t of the prepared buffer holds x[b * xbs + t * ldx + 0..n_in).

// Q8: one warp per 32-element block, rounding as ggml's CPU quantize_row_q8_0 (AVX path): id = 127 / amax,
// q = round-half-even(x * id), stored scale = fp16(amax / 127). qs [C][n_in] int8, then d [C][n_in / 32] float.
__global__ void quantize_q8_kernel(const float * __restrict__ x, int8_t * __restrict__ qs, float * __restrict__ d,
                                   int n_in, int T, int C, int64_t ldx, int64_t xbs) {
    const int64_t blk = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int nb = n_in / 32;
    if (blk >= (int64_t) C * nb) return;
    const int lane = threadIdx.x & 31;
    const int c = (int) (blk / nb), kb = (int) (blk % nb);
    const int b = c / T, t = c % T;
    const float v = x[b * xbs + t * ldx + kb * 32 + lane];
    const float amax = q8_group_amax<32>(fabsf(v));
    qs[(int64_t) c * n_in + kb * 32 + lane] = q8_quant(v, q8_inv_scale(amax));
    if (lane == 0) d[(int64_t) c * nb + kb] = q8_scale(amax);
}

__global__ void to_f16_kernel(const float * __restrict__ x, __half * __restrict__ y, int n_in, int T, int64_t total,
                              int64_t ldx, int64_t xbs) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t c = i / n_in;
        const int k = (int) (i % n_in);
        const int b = (int) (c / T), t = (int) (c % T);
        y[i] = __float2half_rn(x[b * xbs + t * ldx + k]);
    }
}

__device__ __forceinline__ float bf16_to_f32(uint16_t v) { return __uint_as_float((uint32_t) v << 16); }

// Rows [r0, r0 + rows) of the flattened [batch * n_out] row space -> F16 [rows][n_in]. `aux` = Q8_0R scales.
template <int TYPE>
__global__ void dequant_f16_kernel(const uint8_t * __restrict__ w, const __half * __restrict__ aux,
                                   __half * __restrict__ y, int n_in, int64_t r0, int64_t total) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t r = r0 + i / n_in;
        const int k = (int) (i % n_in);
        float v;
        if constexpr (TYPE == kTypeQ8_0) {
            const uint8_t * blk = w + r * ((int64_t) n_in / 32 * kQ8Block) + (k / 32) * kQ8Block;
            v = __half2float(*reinterpret_cast<const __half *>(blk)) * (float) (int8_t) blk[2 + (k & 31)];
        } else if constexpr (TYPE == kTypeQ8_0R) {
            v = __half2float(aux[r * (n_in / 32) + k / 32]) * (float) reinterpret_cast<const int8_t *>(w)[r * n_in + k];
        } else {   // BF16
            v = bf16_to_f32(reinterpret_cast<const uint16_t *>(w)[r * n_in + k]);
        }
        y[i] = __float2half_rn(v);
    }
}

__global__ void repack_q8_0_kernel(const uint8_t * __restrict__ src, int8_t * __restrict__ qs,
                                   __half * __restrict__ sc, int n_in, int64_t nblocks) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < nblocks * 32;
         i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t blk = i / 32;
        const int j = (int) (i % 32);
        const uint8_t * b = src + blk * kQ8Block;
        qs[i] = (int8_t) b[2 + j];
        if (j == 0) sc[blk] = *reinterpret_cast<const __half *>(b);
    }
}

// ---------------------------------------------------------------------------------------------------------
// GEMV. A block of 128 threads is split into groups of TPR threads; each group owns R consecutive rows and
// walks the reduction dimension in chunks, so every activation load serves R rows. NC columns (tokens) per
// launch, all held in registers.

template <int R, int NC>
__device__ __forceinline__ void reduce_store(float (&acc)[R][NC], int tpr, float * __restrict__ y, int row0,
                                             int n_out, int c0, int64_t ybase, int64_t ldy, float beta) {
    __shared__ float red[kThreads / 32][R * NC];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int span = tpr < 32 ? tpr : 32;
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c)
            for (int o = span >> 1; o > 0; o >>= 1) acc[r][c] += __shfl_xor_sync(0xffffffffu, acc[r][c], o);
    if (tpr <= 32) {
        if ((lane & (tpr - 1)) == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r >= n_out) break;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    float * p = y + ybase + (int64_t) (c0 + c) * ldy + row0 + r;
                    *p = beta != 0.0f ? acc[r][c] + beta * *p : acc[r][c];
                }
            }
        }
        return;
    }
    // tpr == 128: one group per block, sum the four warps in order.
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int c = 0; c < NC; ++c) red[warp][r * NC + c] = acc[r][c];
    }
    __syncthreads();
    if (threadIdx.x < R * NC) {
        const int r = threadIdx.x / NC, c = threadIdx.x % NC;
        if (row0 + r < n_out) {
            const float s = red[0][threadIdx.x] + red[1][threadIdx.x] + red[2][threadIdx.x] + red[3][threadIdx.x];
            float * p = y + ybase + (int64_t) (c0 + c) * ldy + row0 + r;
            *p = beta != 0.0f ? s + beta * *p : s;
        }
    }
}

__device__ __forceinline__ int load_int_b2(const uint8_t * p) {   // 2-byte aligned 32-bit load
    const uint16_t * q = reinterpret_cast<const uint16_t *>(p);
    return (int) ((uint32_t) q[0] | ((uint32_t) q[1] << 16));
}

// Raw GGUF Q8_0 (34-byte blocks, 2-byte aligned): 8-byte chunks, 4 per block.
template <int R, int NC>
__global__ void __launch_bounds__(kThreads) gemv_q8_0_kernel(const uint8_t * __restrict__ w, int64_t w_batch_bytes,
                                                             const int8_t * __restrict__ qs,
                                                             const float * __restrict__ xd, float * __restrict__ y,
                                                             int n_in, int n_out, int T, int c0, int tpr,
                                                             int64_t ldy, int64_t ybs, float beta) {
    const int b = blockIdx.y;
    const int g = threadIdx.x / tpr, gl = threadIdx.x % tpr;
    const int row0 = (blockIdx.x * (kThreads / tpr) + g) * R;
    const int nb = n_in / 32, nchunk = n_in / 8;
    const size_t rb = (size_t) nb * kQ8Block;
    const uint8_t * wrow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wrow[r] = w + b * w_batch_bytes + (size_t) min(row0 + r, n_out - 1) * rb;
    const int8_t * xq = qs + ((int64_t) b * T + c0) * n_in;
    const float * xs = xd + ((int64_t) b * T + c0) * nb;

    float acc[R][NC] = {};
    for (int k = gl; k < nchunk; k += tpr) {
        const int kb = k >> 2, sub = k & 3;
        int xv[NC][2];
        float dx[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int2 v = *reinterpret_cast<const int2 *>(xq + (int64_t) c * n_in + k * 8);
            xv[c][0] = v.x;
            xv[c][1] = v.y;
            dx[c] = xs[(int64_t) c * nb + kb];
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t * blk = wrow[r] + (size_t) kb * kQ8Block;
            const float dw = __half2float(*reinterpret_cast<const __half *>(blk));
            const int w0 = load_int_b2(blk + 2 + sub * 8), w1 = load_int_b2(blk + 6 + sub * 8);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const int s = __dp4a(w1, xv[c][1], __dp4a(w0, xv[c][0], 0));
                acc[r][c] += (float) s * (dw * dx[c]);
            }
        }
    }
    reduce_store<R, NC>(acc, tpr, y, row0, n_out, c0, (int64_t) b * ybs, ldy, beta);
}

// Aligned formats, 16-byte weight chunks: Q8_0R (16 quants, Q8 activations), F16/BF16 (8 values) and F32 (4),
// the float formats on the raw F32 activations (16-byte aligned rows).
// Up to kMaxParts weights sharing the input run in one launch: part p owns blocks [blk_end[p-1], blk_end[p]).
constexpr int kMaxParts = 4;
struct G16Part {
    const uint8_t * w;
    const __half * sc;      // Q8_0R scales
    float * y;
    int n_out, blk_end;
    int64_t ldy;
};
struct G16 {
    G16Part part[kMaxParts];
    int nparts;
    int64_t w_bs;           // bytes per batch matrix (Q8_0R: of the quants); batched launches have one part
    int64_t sc_bs;          // scales per batch matrix
    const int8_t * xq;      // Q8 activations
    const float * xd;
    const float * xf;       // F32 activations
    int64_t ldx, xbs;
    int n_in, T, c0, tpr;
    int64_t ybs;
    float beta;
};

__device__ __forceinline__ void unpack8(const int4 v, int type, float (&f)[8]) {
    const uint32_t u[4] = {(uint32_t) v.x, (uint32_t) v.y, (uint32_t) v.z, (uint32_t) v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (type == kTypeF16) {
            const float2 p = __half22float2(*reinterpret_cast<const __half2 *>(&u[i]));
            f[2 * i] = p.x;
            f[2 * i + 1] = p.y;
        } else {
            f[2 * i] = __uint_as_float(u[i] << 16);
            f[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
        }
    }
}

template <int TYPE, int R, int NC>
__global__ void __launch_bounds__(kThreads) gemv16_kernel(const G16 a) {
    constexpr int ES = TYPE == kTypeF32 ? 4 : (TYPE == kTypeQ8_0R ? 1 : 2);
    constexpr int EPC = 16 / ES;   // elements per chunk
    const int b = blockIdx.y;
    int pi = 0;
    while (pi + 1 < a.nparts && (int) blockIdx.x >= a.part[pi].blk_end) ++pi;
    const G16Part & P = a.part[pi];
    const int blk = blockIdx.x - (pi ? a.part[pi - 1].blk_end : 0);
    const int g = threadIdx.x / a.tpr, gl = threadIdx.x % a.tpr;
    const int row0 = (blk * (kThreads / a.tpr) + g) * R;
    const int nchunk = a.n_in / EPC, nb = a.n_in / 32;
    int rows[R];
    const int4 * wrow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        rows[r] = min(row0 + r, P.n_out - 1);
        wrow[r] = reinterpret_cast<const int4 *>(P.w + b * a.w_bs + (int64_t) rows[r] * a.n_in * ES);
    }
    float acc[R][NC] = {};
    if constexpr (TYPE == kTypeQ8_0R) {
        const int8_t * xq = a.xq + ((int64_t) b * a.T + a.c0) * a.n_in;
        const float * xs = a.xd + ((int64_t) b * a.T + a.c0) * nb;
        const __half * sc = P.sc + b * a.sc_bs;
        for (int k = gl; k < nchunk; k += a.tpr) {
            const int kb = k >> 1;
            int4 wv[R];
            float dw[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                wv[r] = wrow[r][k];
                dw[r] = __half2float(sc[(int64_t) rows[r] * nb + kb]);
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const int4 xv = *reinterpret_cast<const int4 *>(xq + (int64_t) c * a.n_in + k * 16);
                const float dx = xs[(int64_t) c * nb + kb];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int s = __dp4a(wv[r].w, xv.w, __dp4a(wv[r].z, xv.z, __dp4a(wv[r].y, xv.y,
                                  __dp4a(wv[r].x, xv.x, 0))));
                    acc[r][c] += (float) s * (dw[r] * dx);
                }
            }
        }
    } else {
        const float * xb = a.xf + b * a.xbs + (int64_t) a.c0 * a.ldx;
        for (int k = gl; k < nchunk; k += a.tpr) {
            float wf[R][EPC];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int4 v = wrow[r][k];
                if constexpr (TYPE == kTypeF32) {
                    wf[r][0] = __int_as_float(v.x);
                    wf[r][1] = __int_as_float(v.y);
                    wf[r][2] = __int_as_float(v.z);
                    wf[r][3] = __int_as_float(v.w);
                } else {
                    unpack8(v, TYPE, reinterpret_cast<float (&)[8]>(wf[r]));
                }
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                float xv[EPC];
                const float4 * xp = reinterpret_cast<const float4 *>(xb + (int64_t) c * a.ldx + k * EPC);
#pragma unroll
                for (int j = 0; j < EPC / 4; ++j) {
                    const float4 v = xp[j];
                    xv[4 * j] = v.x;
                    xv[4 * j + 1] = v.y;
                    xv[4 * j + 2] = v.z;
                    xv[4 * j + 3] = v.w;
                }
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int j = 0; j < EPC; ++j) acc[r][c] = fmaf(wf[r][j], xv[j], acc[r][c]);
            }
        }
    }
    reduce_store<R, NC>(acc, a.tpr, P.y, row0, P.n_out, a.c0, (int64_t) b * a.ybs, P.ldy, a.beta);
}

template <typename WT>
__device__ __forceinline__ float wload(const WT * p, int type) {
    if constexpr (sizeof(WT) == 4) return *p;
    else return type == kTypeF16 ? __half2float(*reinterpret_cast<const __half *>(p)) : bf16_to_f32(*p);
}

// Float weights on unaligned F32 activations: one element per thread per step (fallback).
template <typename WT, int R, int NC>
__global__ void __launch_bounds__(kThreads) gemv_float_kernel(const WT * __restrict__ w, int type,
                                                              const float * __restrict__ x, float * __restrict__ y,
                                                              int n_in, int n_out, int c0, int tpr, int64_t ldx,
                                                              int64_t xbs, int64_t ldy, int64_t ybs, float beta) {
    const int b = blockIdx.y;
    const int g = threadIdx.x / tpr, gl = threadIdx.x % tpr;
    const int row0 = (blockIdx.x * (kThreads / tpr) + g) * R;
    const WT * wrow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wrow[r] = w + ((int64_t) b * n_out + min(row0 + r, n_out - 1)) * n_in;
    const float * xb = x + b * xbs + (int64_t) c0 * ldx;
    float acc[R][NC] = {};
    for (int k = gl; k < n_in; k += tpr) {
        float xv[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) xv[c] = xb[(int64_t) c * ldx + k];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float wv = wload(wrow[r] + k, type);
#pragma unroll
            for (int c = 0; c < NC; ++c) acc[r][c] = fmaf(wv, xv[c], acc[r][c]);
        }
    }
    reduce_store<R, NC>(acc, tpr, y, row0, n_out, c0, (int64_t) b * ybs, ldy, beta);
}

// Threads per row group: all 128 when the row is long and splits evenly, else a power of two <= 32.
int pick_tpr(int nchunk) {
    if (nchunk >= 256 && nchunk % 128 == 0) return 128;
    int t = 4;
    while (t < 32 && t * 2 <= nchunk) t *= 2;
    return t;
}
// Rows per group: enough blocks to fill the V100 (80 SMs) several times over, then reuse activations.
int pick_rows(int n_out, int batch, int tpr) {
    const int64_t groups = (int64_t) n_out * batch;
    const int gpb = kThreads / tpr;
    if (groups / (4 * gpb) >= 640) return 4;
    if (groups / (2 * gpb) >= 320) return 2;
    return 1;
}

struct Q8View {
    const int8_t * qs;
    const float * d;
};
Q8View q8_view(const void * prepared, int n_in, int C) {
    const int8_t * qs = static_cast<const int8_t *>(prepared);
    return {qs, reinterpret_cast<const float *>(qs + q8_scales_offset((size_t) C, n_in))};
}

enum class Gemv { kQ8Raw, k16, kScalar };

struct GemvPlan {
    Gemv kind;
    int tpr, R;
};
// Tuning overrides (benchmarks only): GLM_MM_TPR=4..128 threads per row group, GLM_MM_R=1|2|4 rows per group.
int env_int(const char * name) {
    const char * v = std::getenv(name);
    return v ? std::atoi(v) : 0;
}
GemvPlan tuned(GemvPlan p) {
    static const int tpr = env_int("GLM_MM_TPR"), r = env_int("GLM_MM_R");
    if (tpr) p.tpr = tpr;
    if (r) p.R = r;
    return p;
}

GemvPlan plan_gemv(const WeightRef & w, const Shape & s, const MatmulInput & in) {
    if (w.type == kTypeQ8_0) {
        const int tpr = pick_tpr(s.n_in / 8);
        return {Gemv::kQ8Raw, tpr, pick_rows(s.n_out, s.batch, tpr)};
    }
    const int es = elem_size(w.type);
    bool aligned = (int64_t) s.n_in * es % 16 == 0 && reinterpret_cast<uintptr_t>(w.data) % 16 == 0;
    if (w.type != kTypeQ8_0R)   // float formats read the raw F32 activations with float4 loads
        aligned = aligned && reinterpret_cast<uintptr_t>(in.x) % 16 == 0 && in.ldx % 4 == 0 &&
                  in.x_batch_stride % 4 == 0;
    if (!aligned) {
        if (w.type == kTypeQ8_0R) fail("Q8_0R weights must be 16-byte aligned");
        const int tpr = pick_tpr(s.n_in);
        return {Gemv::kScalar, tpr, pick_rows(s.n_out, s.batch, tpr)};
    }
    const int nchunk = (int) ((int64_t) s.n_in * es / 16);
    if (in.T <= 2 && nchunk >= 128) {
        // decode: one row per group (measured on the V100, T=1: 4096->8192 44 -> 40 us, 16384->4096 87 -> 74 us,
        // head 664 -> 642 us); a whole block per row only for rows of >= 512 chunks
        const bool block_row = (nchunk >= 512 && nchunk % 128 == 0) || (int64_t) s.n_out * s.batch < 4 * 640;
        return {Gemv::k16, block_row && nchunk % 128 == 0 ? 128 : 32, 1};
    }
    const int tpr = pick_tpr(nchunk);
    return {Gemv::k16, tpr, pick_rows(s.n_out, s.batch, tpr)};
}

template <int R, int NC>
void launch_gemv(const WeightRef * ws, float * const * ys, const int64_t * ldys, int n, const Shape & s,
                 const MatmulInput & in, const GemvPlan & p, int c0, int64_t ybs, float beta, cudaStream_t st) {
    const int gpb = kThreads / p.tpr;
    const Q8View q = q8_view(in.prepared, s.n_in, in.T * in.batch);
    if (p.kind != Gemv::k16) {   // one launch per weight
        for (int i = 0; i < n; ++i) {
            const WeightRef & w = ws[i];
            const int n_out = (int) w.ne[1];
            const dim3 grid((n_out + gpb * R - 1) / (gpb * R), s.batch);
            const uint8_t * wb = static_cast<const uint8_t *>(w.data);
            if (p.kind == Gemv::kQ8Raw)
                gemv_q8_0_kernel<R, NC><<<grid, kThreads, 0, st>>>(wb, (int64_t) n_out * (s.n_in / 32) * kQ8Block,
                                                                   q.qs, q.d, ys[i], s.n_in, n_out, in.T, c0, p.tpr,
                                                                   ldys[i], ybs, beta);
            else if (w.type == kTypeF32)
                gemv_float_kernel<float, R, NC><<<grid, kThreads, 0, st>>>(
                    reinterpret_cast<const float *>(wb), w.type, in.x, ys[i], s.n_in, n_out, c0, p.tpr, in.ldx,
                    in.x_batch_stride, ldys[i], ybs, beta);
            else
                gemv_float_kernel<uint16_t, R, NC><<<grid, kThreads, 0, st>>>(
                    reinterpret_cast<const uint16_t *>(wb), w.type, in.x, ys[i], s.n_in, n_out, c0, p.tpr, in.ldx,
                    in.x_batch_stride, ldys[i], ybs, beta);
        }
        return;
    }
    G16 a{};
    int blocks = 0;
    for (int i = 0; i < n; ++i) {
        const WeightRef & w = ws[i];
        const int n_out = (int) w.ne[1];
        G16Part & P = a.part[i];
        P.w = static_cast<const uint8_t *>(w.data);
        if (w.type == kTypeQ8_0R)
            P.sc = reinterpret_cast<const __half *>(P.w + q8r_scale_offset((int64_t) n_out * s.batch, s.n_in));
        P.y = ys[i];
        P.n_out = n_out;
        P.ldy = ldys[i];
        blocks += (n_out + gpb * R - 1) / (gpb * R);
        P.blk_end = blocks;
    }
    a.nparts = n;
    a.w_bs = (int64_t) s.n_out * s.n_in * elem_size(ws[0].type);
    a.sc_bs = (int64_t) s.n_out * (s.n_in / 32);
    a.xq = q.qs;
    a.xd = q.d;
    a.xf = in.x;
    a.ldx = in.ldx;
    a.xbs = in.x_batch_stride;
    a.n_in = s.n_in;
    a.T = in.T;
    a.c0 = c0;
    a.tpr = p.tpr;
    a.ybs = ybs;
    a.beta = beta;
    const dim3 grid(blocks, s.batch);
    switch (ws[0].type) {
        case kTypeQ8_0R: gemv16_kernel<kTypeQ8_0R, R, NC><<<grid, kThreads, 0, st>>>(a); break;
        case kTypeF16: gemv16_kernel<kTypeF16, R, NC><<<grid, kThreads, 0, st>>>(a); break;
        case kTypeBF16: gemv16_kernel<kTypeBF16, R, NC><<<grid, kThreads, 0, st>>>(a); break;
        default: gemv16_kernel<kTypeF32, R, NC><<<grid, kThreads, 0, st>>>(a); break;
    }
}

template <int R>
void dispatch_cols(const WeightRef * ws, float * const * ys, const int64_t * ldys, int n, const Shape & s,
                   const MatmulInput & in, const GemvPlan & p, int c0, int nc, int64_t ybs, float beta,
                   cudaStream_t st) {
    switch (nc) {
        case 1: launch_gemv<R, 1>(ws, ys, ldys, n, s, in, p, c0, ybs, beta, st); break;
        case 2: launch_gemv<R, 2>(ws, ys, ldys, n, s, in, p, c0, ybs, beta, st); break;
        case 3: launch_gemv<R, 3>(ws, ys, ldys, n, s, in, p, c0, ybs, beta, st); break;
        case 4: launch_gemv<R, 4>(ws, ys, ldys, n, s, in, p, c0, ybs, beta, st); break;
        case 5: launch_gemv<R, 5>(ws, ys, ldys, n, s, in, p, c0, ybs, beta, st); break;
        case 6: launch_gemv<R, 6>(ws, ys, ldys, n, s, in, p, c0, ybs, beta, st); break;
        case 7: launch_gemv<R, 7>(ws, ys, ldys, n, s, in, p, c0, ybs, beta, st); break;
        case 8: launch_gemv<R, 8>(ws, ys, ldys, n, s, in, p, c0, ybs, beta, st); break;
        default: fail("bad column count");
    }
}

// `s` describes ws[0]; every weight shares its type, n_in and batch (several weights only with batch 1).
void run_gemv(const WeightRef * ws, float * const * ys, const int64_t * ldys, int n, const Shape & s,
              const MatmulInput & in, int64_t ybs, float beta, cudaStream_t st) {
    int max_out = 0;
    for (int i = 0; i < n; ++i) max_out = std::max(max_out, (int) ws[i].ne[1]);
    for (int c0 = 0; c0 < in.T; c0 += 8) {
        const int nc = std::min(8, in.T - c0);
        GemvPlan p = plan_gemv(ws[0], s, in);
        // Several tokens on long rows: one warp per 4 rows keeps more independent loads in flight.
        if (p.kind == Gemv::k16 && nc >= 5 && p.tpr == 128 && (int64_t) max_out * s.batch >= 2048) p = {p.kind, 32, 4};
        p = tuned(p);
        if (p.R == 4) dispatch_cols<4>(ws, ys, ldys, n, s, in, p, c0, nc, ybs, beta, st);
        else if (p.R == 2) dispatch_cols<2>(ws, ys, ldys, n, s, in, p, c0, nc, ybs, beta, st);
        else dispatch_cols<1>(ws, ys, ldys, n, s, in, p, c0, nc, ybs, beta, st);
    }
    ck(cudaGetLastError(), "gemv launch");
}

// ---------------------------------------------------------------------------------------------------------
// GEMM (cuBLAS), column-major view: y^T [n_out x T] = W [n_in x n_out]^T . x^T [n_in x T], ldc = ldy.

struct Blas {
    cublasHandle_t h = nullptr;
    void * ws = nullptr;
};
Blas & blas() {
    thread_local Blas b[16];
    int dev = 0;
    ck(cudaGetDevice(&dev), "cudaGetDevice");
    Blas & r = b[dev & 15];
    if (!r.h) {
        ck(cublasCreate(&r.h), "cublasCreate");
        const size_t ws = 32u << 20;
        ck(cudaMalloc(&r.ws, ws), "cuBLAS workspace");
        ck(cublasSetWorkspace(r.h, r.ws, ws), "cublasSetWorkspace");
        ck(cublasSetMathMode(r.h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    }
    return r;
}

void gemm(cudaDataType ta, const void * A, int64_t strideA, const void * B, int64_t ldb, int64_t strideB, float * C,
          int64_t ldc, int64_t strideC, int m, int n, int k, int count, float beta, cudaStream_t st) {
    Blas & b = blas();
    ck(cublasSetStream(b.h, st), "cublasSetStream");
    const float alpha = 1.0f;
    if (count == 1)
        ck(cublasGemmEx(b.h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, ta, k, B, ta, (int) ldb, &beta, C,
                        CUDA_R_32F, (int) ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
           "cublasGemmEx");
    else
        ck(cublasGemmStridedBatchedEx(b.h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, ta, k, strideA, B, ta,
                                      (int) ldb, strideB, &beta, C, CUDA_R_32F, (int) ldc, strideC, count,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
           "cublasGemmStridedBatchedEx");
}

int grid_for(int64_t total) { return (int) std::min<int64_t>((total + 255) / 256, 80 * 32); }

void dequant_rows(const WeightRef & w, const Shape & s, int64_t r0, int64_t rows, __half * out, cudaStream_t st) {
    const uint8_t * wb = static_cast<const uint8_t *>(w.data);
    const int64_t total = rows * s.n_in;
    const int g = grid_for(total);
    switch (w.type) {
        case kTypeQ8_0: dequant_f16_kernel<kTypeQ8_0><<<g, 256, 0, st>>>(wb, nullptr, out, s.n_in, r0, total); break;
        case kTypeQ8_0R: {
            const auto * sc = reinterpret_cast<const __half *>(wb + q8r_scale_offset((int64_t) s.n_out * s.batch,
                                                                                     s.n_in));
            dequant_f16_kernel<kTypeQ8_0R><<<g, 256, 0, st>>>(wb, sc, out, s.n_in, r0, total);
            break;
        }
        default: dequant_f16_kernel<kTypeBF16><<<g, 256, 0, st>>>(wb, nullptr, out, s.n_in, r0, total); break;
    }
}

void run_gemm(const WeightRef & w, const Shape & s, const MatmulInput & in, float * y, int64_t ldy, int64_t ybs,
              float beta, void * ws, size_t ws_bytes, cudaStream_t st) {
    const int T = in.T;
    if (w.type == kTypeF32) {   // SGEMM straight on the strided F32 activations
        gemm(CUDA_R_32F, w.data, (int64_t) s.n_out * s.n_in, in.x, in.ldx, in.x_batch_stride, y, ldy, ybs, s.n_out, T,
             s.n_in, s.batch, beta, st);
        return;
    }
    const __half * xh = static_cast<const __half *>(in.prepared);
    const int64_t xh_bs = (int64_t) T * s.n_in;
    if (w.type == kTypeF16) {
        gemm(CUDA_R_16F, w.data, (int64_t) s.n_out * s.n_in, xh, s.n_in, xh_bs, y, ldy, ybs, s.n_out, T, s.n_in,
             s.batch, beta, st);
        return;
    }
    // Q8_0 / Q8_0R / BF16: dequantize to F16 in slices that fit the scratch.
    const int64_t cap_rows = (int64_t) (ws_bytes / 2) / s.n_in;
    if (cap_rows < 1) fail("scratch too small for one F16 weight row");
    __half * wh = static_cast<__half *>(ws);
    if (cap_rows >= (int64_t) s.n_out && s.batch > 1) {   // whole matrices: batched GEMM over as many as fit
        const int per = (int) std::min<int64_t>(s.batch, cap_rows / s.n_out);
        for (int b0 = 0; b0 < s.batch; b0 += per) {
            const int nbt = std::min(per, s.batch - b0);
            dequant_rows(w, s, (int64_t) b0 * s.n_out, (int64_t) nbt * s.n_out, wh, st);
            gemm(CUDA_R_16F, wh, (int64_t) s.n_out * s.n_in, xh + b0 * xh_bs, s.n_in, xh_bs, y + b0 * ybs, ldy, ybs,
                 s.n_out, T, s.n_in, nbt, beta, st);
        }
    } else {
        for (int b = 0; b < s.batch; ++b)
            for (int64_t r0 = 0; r0 < s.n_out; r0 += cap_rows) {
                const int n = (int) std::min<int64_t>(cap_rows, s.n_out - r0);
                dequant_rows(w, s, (int64_t) b * s.n_out + r0, n, wh, st);
                gemm(CUDA_R_16F, wh, 0, xh + b * xh_bs, s.n_in, 0, y + b * ybs + r0, ldy, 0, n, T, s.n_in, 1, beta,
                     st);
            }
    }
    ck(cudaGetLastError(), "dequant launch");
}

bool use_gemv(int T) { return T <= kMatmulGemvMaxT; }

}  // namespace

size_t matmul_prepare_bytes(int n_in, int T, int batch) {
    const size_t C = (size_t) T * batch;
    if (use_gemv(T)) return align256(C * n_in) + align256(C * (n_in / 32) * sizeof(float));
    return align256(C * n_in * 2);
}

size_t matmul_run_bytes(const WeightRef & w, int T) {
    const Shape s = shape_of(w);
    if (use_gemv(T) || w.type == kTypeF32 || w.type == kTypeF16) return 0;
    return align256((size_t) s.batch * s.n_out * s.n_in * 2);
}

size_t matmul_scratch_bytes(const WeightRef & w, int T) {
    const Shape s = shape_of(w);
    return matmul_prepare_bytes(s.n_in, T, s.batch) + matmul_run_bytes(w, T);
}

MatmulInput matmul_prepare(const float * x, int n_in, int T, int batch, int64_t ldx, int64_t xbs, void * scratch,
                           cudaStream_t stream) {
    if (T <= 0 || n_in <= 0 || batch <= 0) fail("empty input");
    MatmulInput in;
    in.x = x;
    in.n_in = n_in;
    in.T = T;
    in.batch = batch;
    in.ldx = ldx > 0 ? ldx : n_in;
    in.x_batch_stride = xbs > 0 ? xbs : (int64_t) T * in.ldx;
    in.prepared = scratch;
    const int C = T * batch;
    if (use_gemv(T)) {
        if (n_in % 32) return in;   // only float weights can take this input (Q8_0 needs n_in % 32 == 0)
        in.kind = kQ8;
        const Q8View q = q8_view(scratch, n_in, C);
        const int64_t blocks = (int64_t) C * (n_in / 32);
        quantize_q8_kernel<<<(unsigned) ((blocks + 7) / 8), 256, 0, stream>>>(
            x, const_cast<int8_t *>(q.qs), const_cast<float *>(q.d), n_in, T, C, in.ldx, in.x_batch_stride);
    } else {
        in.kind = kF16;
        const int64_t total = (int64_t) C * n_in;
        to_f16_kernel<<<grid_for(total), 256, 0, stream>>>(x, static_cast<__half *>(scratch), n_in, T, total, in.ldx,
                                                            in.x_batch_stride);
    }
    ck(cudaGetLastError(), "prepare launch");
    return in;
}

bool matmul_input_is_q8(int n_in, int T) { return T > 0 && use_gemv(T) && n_in % 32 == 0; }

MatmulInput matmul_input_q8(const float * x, int n_in, int T, const void * prepared, int batch, int64_t ldx,
                            int64_t xbs) {
    if (!matmul_input_is_q8(n_in, T)) fail("matmul_input_q8: this (n_in, T) is not on the Q8 GEMV path");
    if (batch < 1) fail("matmul_input_q8: batch < 1");
    MatmulInput in;
    in.x = x;
    in.n_in = n_in;
    in.T = T;
    in.batch = batch;
    in.ldx = ldx > 0 ? ldx : n_in;
    in.x_batch_stride = xbs > 0 ? xbs : (int64_t) T * in.ldx;
    in.prepared = const_cast<void *>(prepared);
    in.kind = kQ8;
    return in;
}

void matmul_run(const WeightRef & w, const MatmulInput & in, float * y, void * w_scratch, size_t w_scratch_bytes,
                cudaStream_t stream, int64_t ldy, int64_t ybs, float beta) {
    const Shape s = shape_of(w);
    if (s.n_in != in.n_in || s.batch != in.batch) fail("input does not match the weight (n_in or batch)");
    if (ldy <= 0) ldy = s.n_out;
    if (ybs <= 0) ybs = (int64_t) in.T * ldy;
    if (use_gemv(in.T)) {
        if (is_q8(w.type) && in.kind != kQ8) fail("Q8_0 weight without a Q8 prepared input");
        run_gemv(&w, &y, &ldy, 1, s, in, ybs, beta, stream);
    } else {
        run_gemm(w, s, in, y, ldy, ybs, beta, w_scratch, w_scratch_bytes, stream);
    }
}

void matmul_run_multi(const WeightRef * ws, float * const * ys, const int64_t * ldys, int n, const MatmulInput & in,
                      void * w_scratch, size_t w_scratch_bytes, cudaStream_t stream) {
    if (n <= 0) return;
    const Shape s = shape_of(ws[0]);
    int64_t ld[kMaxParts];
    for (int i = 0; i < n; ++i) {
        const Shape si = shape_of(ws[i]);
        ld[i % kMaxParts] = ldys && ldys[i] > 0 ? ldys[i] : si.n_out;
    }
    const bool fused = use_gemv(in.T) && n <= kMaxParts && s.batch == 1 && in.batch == 1;
    for (int i = 0; fused && i < n; ++i)
        if (ws[i].type != ws[0].type || ws[i].ne[0] != ws[0].ne[0] || ws[i].ne[2] != 1) fail("multi: weights differ");
    if (!fused || plan_gemv(ws[0], s, in).kind != Gemv::k16) {
        for (int i = 0; i < n; ++i)
            matmul_run(ws[i], in, ys[i], w_scratch, w_scratch_bytes, stream, ldys ? ldys[i] : 0);
        return;
    }
    if (s.n_in != in.n_in) fail("input does not match the weights (n_in)");
    if (is_q8(ws[0].type) && in.kind != kQ8) fail("Q8_0 weight without a Q8 prepared input");
    run_gemv(ws, ys, ld, n, s, in, 0, 0.0f, stream);
}

void glm_matmul(const WeightRef & w, const float * x, float * y, int T, void * scratch, size_t scratch_bytes,
                cudaStream_t stream, const MatmulStrides & st) {
    const Shape s = shape_of(w);
    const size_t pb = matmul_prepare_bytes(s.n_in, T, s.batch);
    if (scratch_bytes < pb) fail("scratch smaller than the prepared activations");
    MatmulInput in;
    if (use_gemv(T) && !is_q8(w.type)) {   // float GEMV reads the raw x: no quantize launch
        if (T <= 0 || s.n_in <= 0) fail("empty input");
        in.x = x;
        in.n_in = s.n_in;
        in.T = T;
        in.batch = s.batch;
        in.ldx = st.ldx > 0 ? st.ldx : s.n_in;
        in.x_batch_stride = st.x_batch_stride > 0 ? st.x_batch_stride : (int64_t) T * in.ldx;
        in.prepared = scratch;
    } else {
        in = matmul_prepare(x, s.n_in, T, s.batch, st.ldx, st.x_batch_stride, scratch, stream);
    }
    matmul_run(w, in, y, static_cast<char *>(scratch) + pb, scratch_bytes - pb, stream, st.ldy, st.y_batch_stride,
               st.beta);
}

size_t matmul_repack_bytes(const WeightRef & w) {
    const Shape s = shape_of(w);
    if (w.type != kTypeQ8_0) fail("repack: only Q8_0 is repacked");
    return q8r_bytes((int64_t) s.n_out * s.batch, s.n_in);
}

WeightRef matmul_repack(const WeightRef & w, void * dst, cudaStream_t stream) {
    const Shape s = shape_of(w);
    if (w.type != kTypeQ8_0) fail("repack: only Q8_0 is repacked");
    if (reinterpret_cast<uintptr_t>(dst) % 256) fail("repack: destination must be 256-byte aligned");
    const int64_t rows = (int64_t) s.n_out * s.batch, nblocks = rows * (s.n_in / 32);
    int8_t * qs = static_cast<int8_t *>(dst);
    __half * sc = reinterpret_cast<__half *>(static_cast<char *>(dst) + q8r_scale_offset(rows, s.n_in));
    repack_q8_0_kernel<<<grid_for(nblocks * 32), 256, 0, stream>>>(static_cast<const uint8_t *>(w.data), qs, sc,
                                                                   s.n_in, nblocks);
    ck(cudaGetLastError(), "repack launch");
    WeightRef r = w;
    r.data = dst;
    r.type = kTypeQ8_0R;
    r.nbytes = q8r_bytes(rows, s.n_in);
    return r;
}

void matmul_init() { (void) blas(); }

}  // namespace strata::glm
