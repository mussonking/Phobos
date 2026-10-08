// src/glm/ops/trunk_ops.cu - see include/strata/glm/ops.hpp.
//
// Float order follows the ggml CPU ops the oracle ran (dsv4_hc_comb/pre/post, rms_norm then mul, swiglu_clamp):
// explicit __fmul_rn/__fadd_rn where ggml does a separate mul then add, sequential sums where it sums
// sequentially, so the result stays within float rounding of the reference.
#include "strata/glm/ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "strata/glm/matmul.hpp"
#include "strata/glm/q8_block.cuh"

namespace strata::glm {
namespace {

constexpr int kTypeF32 = 0, kTypeF16 = 1, kTypeQ8_0 = 8, kTypeBF16 = 30;
constexpr int kHc = 4, kMix = 24, kEmbd = 4096;
constexpr int kMixThreads = kHc * kEmbd / kHcSplit;   // one thread per flattened stream feature
constexpr int kRowThreads = 256;                // per-token row kernels: 16 features per thread
constexpr int kTileT = 32;                      // tokens per block of the token-tiled mix (prefill)
constexpr int kTileThreads = 256;
#ifndef GLM_HC_MIX_TILE_MIN
#define GLM_HC_MIX_TILE_MIN 64
#endif
constexpr int kMixTileMin = GLM_HC_MIX_TILE_MIN;   // T from which hc_init / hc_post use the token-tiled mix

[[noreturn]] void fail(const std::string & what) { throw std::runtime_error("glm ops: " + what); }

void check_geometry(const Geometry & g) {
    if (g.hc != kHc || g.hc_mix != kMix || g.n_embd != kEmbd) fail("geometry differs from hc=4, mix=24, D=4096");
}

void check_f32(const WeightRef & w, int64_t n, const char * what) {
    if (w.type != kTypeF32 || w.ne[0] != n || !w.data) fail(std::string(what) + " must be F32 [" + std::to_string(n) + "]");
}

// hc_fn as seen by the mix kernel: element (r, c) of the [24][16384] matrix.
struct FnView {
    const uint8_t * w = nullptr;
    const __half * sc = nullptr;   // Q8_0R scales
    int type = -1;
    int n_in = 0;
};

FnView fn_view(const HcBranch & b) {
    const WeightRef & w = b.fn;
    if (w.ne[0] != kHc * kEmbd || w.ne[1] != kMix || w.ne[2] != 1 || !w.data) fail("hc_fn must be [16384 -> 24]");
    check_f32(b.base, kMix, "hc_base");
    check_f32(b.scale, 3, "hc_scale");
    FnView v;
    v.w = static_cast<const uint8_t *>(w.data);
    v.type = w.type;
    v.n_in = (int) w.ne[0];
    if (w.type == kTypeQ8_0R) {
        const size_t off = ((size_t) kMix * v.n_in + 255) & ~size_t(255);
        v.sc = reinterpret_cast<const __half *>(v.w + off);
    } else if (w.type != kTypeQ8_0 && w.type != kTypeBF16 && w.type != kTypeF16 && w.type != kTypeF32) {
        fail("hc_fn type " + std::to_string(w.type) + " not supported");
    }
    return v;
}

// WT is the weight type as a template parameter: a runtime switch per element keeps the compiler from batching the
// 96 independent weight loads of a thread (they serialize, ~45 us per decode call instead of a few).
template <int WT>
__device__ __forceinline__ float fn_at(const FnView & v, int r, int c) {
    switch (WT) {
        case kTypeQ8_0R:
            return __half2float(v.sc[r * (v.n_in / 32) + c / 32]) *
                   (float) reinterpret_cast<const int8_t *>(v.w)[(int64_t) r * v.n_in + c];
        case kTypeQ8_0: {
            const uint8_t * blk = v.w + ((int64_t) r * (v.n_in / 32) + c / 32) * 34;
            return __half2float(*reinterpret_cast<const __half *>(blk)) * (float) (int8_t) blk[2 + (c & 31)];
        }
        case kTypeBF16:
            return __uint_as_float((uint32_t) reinterpret_cast<const uint16_t *>(v.w)[(int64_t) r * v.n_in + c] << 16);
        case kTypeF16:
            return __half2float(reinterpret_cast<const __half *>(v.w)[(int64_t) r * v.n_in + c]);
        default:
            return reinterpret_cast<const float *>(v.w)[(int64_t) r * v.n_in + c];
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Block-wide sum for kRowThreads threads; every thread gets the result.
__device__ __forceinline__ float block_sum(float v, float * red) {
    v = warp_sum(v);
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    __syncthreads();   // red may still be read by a previous call
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < kRowThreads / 32; ++i) s += red[i];
    return s;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

// grid (kHcSplit, T), kMixThreads threads; thread owns one flattened feature c = dst * 4096 + i of the streams.
// MODE 0: init (x_out[dst] = e), MODE 1: post-mix x_in -> x_out. With NEXT, writes the next branch's mix partials
// part[t][r][block] (r < 24: W_r . x_out, r = 24: sum of squares).
template <int MODE, bool NEXT, int WT>
__global__ void __launch_bounds__(kMixThreads)
hc_mix_kernel(const float * __restrict__ src, const float * __restrict__ src2, const float * __restrict__ hcp,
              const float * __restrict__ x_in, float * __restrict__ x_out, FnView fn, float * __restrict__ part) {
    const int t = blockIdx.y;
    const int c = blockIdx.x * kMixThreads + threadIdx.x;
    const int dst = c / kEmbd, i = c % kEmbd;
    float v;
    if constexpr (MODE == 0) {
        v = src[(int64_t) t * kEmbd + i];
    } else {
        float b = src[(int64_t) t * kEmbd + i];
        if (src2) b = __fadd_rn(b, src2[(int64_t) t * kEmbd + i]);
        const float * h = hcp + t * kMix;
        const float * xt = x_in + (int64_t) t * kHc * kEmbd + i;
        float r[kHc];
#pragma unroll
        for (int s = 0; s < kHc; ++s) r[s] = xt[s * kEmbd];
        v = __fmul_rn(b, h[4 + dst]);
#pragma unroll
        for (int s = 0; s < kHc; ++s) v = __fadd_rn(v, __fmul_rn(r[s], h[8 + dst + 4 * s]));
    }
    x_out[(int64_t) t * kHc * kEmbd + c] = v;
    if constexpr (NEXT) {
        float acc[kMix + 1];
#pragma unroll
        for (int r = 0; r < kMix; ++r) acc[r] = fn_at<WT>(fn, r, c) * v;
        acc[kMix] = v * v;
        __shared__ float red[kMixThreads / 32][kMix + 1];
        const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
#pragma unroll
        for (int r = 0; r <= kMix; ++r) {
            const float s = warp_sum(acc[r]);
            if (lane == 0) red[warp][r] = s;
        }
        __syncthreads();
        if (threadIdx.x <= kMix) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < kMixThreads / 32; ++w) s += red[w][threadIdx.x];
            part[((int64_t) t * (kMix + 1) + threadIdx.x) * kHcSplit + blockIdx.x] = s;
        }
    }
}

// One partial of hc_mix_kernel in its exact float order: the products rounded, each warp's 32 summed by the xor
// butterfly, then the warps' sums in order from 0. Lane 0's butterfly is the balanced binary tree over the lanes in
// bit-reversed order ((x0 + x16) + (x8 + x24)) + ...; it is evaluated here with a stack of complete-subtree sums
// (6 registers) instead of a 32-element array.
template <class Term>
__device__ __forceinline__ float mix_block_sum(Term term) {
    float s = 0.0f;
#pragma unroll 1
    for (int w = 0; w < kMixThreads / 32; ++w) {
        float st[6];   // st[k]: the sum of the last complete subtree of 2^k leaves
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            float x = term(w * 32 + (int) (__brev((unsigned) i) >> 27));
            int k = 0;
#pragma unroll
            for (int n = i + 1; (n & 1) == 0; n >>= 1, ++k) x = __fadd_rn(st[k], x);
            st[k] = x;
        }
        s = __fadd_rn(s, st[5]);
    }
    return s;
}

// Token-tiled hc_mix_kernel for large T (prefill): grid (kHcSplit, ceil(T / kTileT)), kTileThreads threads; the same
// outputs, bit for bit, for kTileT tokens of one block of kMixThreads features (a token's result does not depend on
// T or on the kernel). The block of hc_fn is decoded once into shared memory and shared by the tokens; each partial
// is summed by one thread in hc_mix_kernel's order (mix_block_sum) instead of 25 warp reductions per element.
template <int MODE, bool NEXT, int WT>
__global__ void __launch_bounds__(kTileThreads, 3)   // 3 blocks per SM (shared memory allows 3)
hc_mix_tiled_kernel(const float * __restrict__ src, const float * __restrict__ src2, const float * __restrict__ hcp,
                    const float * __restrict__ x_in, float * __restrict__ x_out, FnView fn, float * __restrict__ part,
                    int T) {
    __shared__ float vs[NEXT ? kTileT : 1][kMixThreads + 1];   // +1: the dot-product reads are bank-conflict free
    __shared__ float fs[NEXT ? kMix : 1][kMixThreads + 1];
    const int t0 = blockIdx.y * kTileT, c0 = blockIdx.x * kMixThreads;
    if constexpr (NEXT) {
        for (int k = threadIdx.x; k < kMix * kMixThreads; k += kTileThreads)
            fs[k / kMixThreads][k % kMixThreads] = fn_at<WT>(fn, k / kMixThreads, c0 + k % kMixThreads);
    }
    const int j = threadIdx.x % kMixThreads;
    const int c = c0 + j, dst = c / kEmbd, i = c % kEmbd;
    for (int tt = threadIdx.x / kMixThreads; tt < kTileT && t0 + tt < T; tt += kTileThreads / kMixThreads) {
        const int t = t0 + tt;
        float v;
        if constexpr (MODE == 0) {
            v = src[(int64_t) t * kEmbd + i];
        } else {
            float b = src[(int64_t) t * kEmbd + i];
            if (src2) b = __fadd_rn(b, src2[(int64_t) t * kEmbd + i]);
            const float * h = hcp + t * kMix;
            const float * xt = x_in + (int64_t) t * kHc * kEmbd + i;
            float r[kHc];
#pragma unroll
            for (int s = 0; s < kHc; ++s) r[s] = xt[s * kEmbd];
            v = __fmul_rn(b, h[4 + dst]);
#pragma unroll
            for (int s = 0; s < kHc; ++s) v = __fadd_rn(v, __fmul_rn(r[s], h[8 + dst + 4 * s]));
        }
        x_out[(int64_t) t * kHc * kEmbd + c] = v;
        if constexpr (NEXT) vs[tt][j] = v;
    }
    if constexpr (NEXT) {
        __syncthreads();
        // thread = (token tt, row group rg): rows rg, rg + 8, rg + 16, and the sum of squares (written by rg 0)
        const int tt = threadIdx.x / 8, rg = threadIdx.x % 8, t = t0 + tt;
        if (t >= T) return;
        const float * v = vs[tt];
        float * p = part + (int64_t) t * (kMix + 1) * kHcSplit + blockIdx.x;
        for (int r = rg; r < kMix; r += 8) {
            const float * f = fs[r];
            p[(int64_t) r * kHcSplit] = mix_block_sum([&](int k) { return __fmul_rn(f[k], v[k]); });
        }
        if (rg == 0) p[(int64_t) kMix * kHcSplit] = mix_block_sum([&](int k) { return __fmul_rn(v[k], v[k]); });
    }
}

// 1/s with the MUFU approximation plus one Newton step (within 1 ulp): __frcp_rn is a slow multi-instruction path
// on sm_70 and 160 of them per Sinkhorn sit on the critical path.
__device__ __forceinline__ float rcp_fast(float s) {
    const float r = __fdividef(1.0f, s);
    return fmaf(r, fmaf(-s, r, 1.0f), r);
}

// The 4x4 combine of one token on every lane (no shuffles: each lane keeps all 16 values; sums in the CPU order).
__device__ __forceinline__ void sinkhorn16(float (&c)[16], float eps, int iters) {
    // softmax over dst for each src (c[dst + 4 src])
#pragma unroll
    for (int src = 0; src < 4; ++src) {
        float m = -INFINITY;
#pragma unroll
        for (int d = 0; d < 4; ++d) m = fmaxf(m, c[d + 4 * src]);
        float s = 0.0f;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            c[d + 4 * src] = expf(c[d + 4 * src] - m);
            s += c[d + 4 * src];
        }
        const float inv = rcp_fast(s);
#pragma unroll
        for (int d = 0; d < 4; ++d) c[d + 4 * src] = __fadd_rn(__fmul_rn(c[d + 4 * src], inv), eps);
    }
    auto norm_cols = [&]() {   // fixed dst, sum over src
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            float s = eps;
#pragma unroll
            for (int k = 0; k < 4; ++k) s += c[d + 4 * k];
            const float inv = rcp_fast(s);
#pragma unroll
            for (int k = 0; k < 4; ++k) c[d + 4 * k] *= inv;
        }
    };
    auto norm_rows = [&]() {   // fixed src, sum over dst
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float s = eps;
#pragma unroll
            for (int d = 0; d < 4; ++d) s += c[d + 4 * k];
            const float inv = rcp_fast(s);
#pragma unroll
            for (int d = 0; d < 4; ++d) c[d + 4 * k] *= inv;
        }
    };
    norm_cols();
    for (int it = 1; it < iters; ++it) {
        norm_rows();
        norm_cols();
    }
}

// One thread per token: hcp[t][8..24) combine logits -> Sinkhorn combine, in place.
__global__ void hc_comb_kernel(float * __restrict__ hcp, int T, float eps, int iters) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float * h = hcp + (int64_t) t * kMix + 8;
    float c[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) c[k] = h[k];
    sinkhorn16(c, eps, iters);
#pragma unroll
    for (int k = 0; k < 16; ++k) h[k] = c[k];
}

// grid T, kRowThreads threads.
__global__ void __launch_bounds__(kRowThreads)
hc_pre_kernel(const float * __restrict__ x, const float * __restrict__ part, const float * __restrict__ base,
              const float * __restrict__ scale, const float * __restrict__ norm_w, float * __restrict__ hcp,
              float * __restrict__ y, int8_t * __restrict__ yq, float * __restrict__ yd, float hc_eps,
              float rms_eps) {
    const int t = blockIdx.x;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    __shared__ float mix[kMix + 1];
    __shared__ float pre[kHc];
    __shared__ float red[kRowThreads / 32];
    for (int r = warp; r <= kMix; r += kRowThreads / 32) {
        const float * p = part + ((int64_t) t * (kMix + 1) + r) * kHcSplit;
        float s = 0.0f;
#pragma unroll
        for (int k = lane; k < kHcSplit; k += 32) s += p[k];
        s = warp_sum(s);
        if (lane == 0) mix[r] = s;
    }
    __syncthreads();
    if (warp == 0) {
        const float rinv = 1.0f / sqrtf(mix[kMix] / (float) (kHc * kEmbd) + rms_eps);
        float * h = hcp + t * kMix;
        if (lane < 8) {   // pre (lanes 0-3), post (4-7)
            const float a = __fadd_rn(__fmul_rn(mix[lane] * rinv, scale[lane / 4]), base[lane]);
            const float sg = sigmoidf_(a);
            const float out = lane < 4 ? sg + hc_eps : 2.0f * sg;
            h[lane] = out;
            if (lane < 4) pre[lane] = out;
        } else if (lane < 24) {   // combine logits; hc_comb runs the Sinkhorn on them
            h[lane] = __fadd_rn(__fmul_rn(mix[lane] * rinv, scale[2]), base[lane]);
        }
    }
    __syncthreads();
    // stream sum + RMSNorm: thread owns float4 groups k * kRowThreads + tid
    constexpr int kPer4 = kEmbd / 4 / kRowThreads;
    const float4 * xt = reinterpret_cast<const float4 *>(x + (int64_t) t * kHc * kEmbd);
    float4 b[kPer4];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < kPer4; ++k) {
        const int q = k * kRowThreads + threadIdx.x;
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int h = 0; h < kHc; ++h) {
            const float4 v = xt[h * (kEmbd / 4) + q];
            s.x = fmaf(v.x, pre[h], s.x);
            s.y = fmaf(v.y, pre[h], s.y);
            s.z = fmaf(v.z, pre[h], s.z);
            s.w = fmaf(v.w, pre[h], s.w);
        }
        b[k] = s;
        ss = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, ss))));
    }
    ss = block_sum(ss, red);
    const float r = 1.0f / sqrtf(ss / (float) kEmbd + rms_eps);
    const float4 * w4 = reinterpret_cast<const float4 *>(norm_w);
    float4 * y4 = reinterpret_cast<float4 *>(y + (int64_t) t * kEmbd);
#pragma unroll
    for (int k = 0; k < kPer4; ++k) {
        const int q = k * kRowThreads + threadIdx.x;
        const float4 w = w4[q];
        const float4 o = make_float4(__fmul_rn(b[k].x * r, w.x), __fmul_rn(b[k].y * r, w.y),
                                     __fmul_rn(b[k].z * r, w.z), __fmul_rn(b[k].w * r, w.w));
        y4[q] = o;
        if (yq) {   // Q8 prepared input: 8 consecutive lanes own one 32-block
            const float amax = q8_group_amax<8>(fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w))));
            const float id = q8_inv_scale(amax);
            reinterpret_cast<char4 *>(yq + (int64_t) t * kEmbd)[q] =
                make_char4(q8_quant(o.x, id), q8_quant(o.y, id), q8_quant(o.z, id), q8_quant(o.w, id));
            if ((threadIdx.x & 7) == 0) yd[(int64_t) t * (kEmbd / 32) + q / 8] = q8_scale(amax);
        }
    }
}

__global__ void __launch_bounds__(kRowThreads)
hc_head_kernel(const float * __restrict__ x, const float * __restrict__ norm_w, float * __restrict__ y,
               int8_t * __restrict__ yq, float * __restrict__ yd, float rms_eps) {
    const int t = blockIdx.x;
    __shared__ float red[kRowThreads / 32];
    constexpr int kPer = kEmbd / kRowThreads;
    const float * xt = x + (int64_t) t * kHc * kEmbd;
    float m[kPer];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int i = k * kRowThreads + threadIdx.x;
        float s = xt[i];
#pragma unroll
        for (int h = 1; h < kHc; ++h) s = __fadd_rn(s, xt[h * kEmbd + i]);
        m[k] = s * (1.0f / kHc);
        ss = fmaf(m[k], m[k], ss);
    }
    ss = block_sum(ss, red);
    const float r = 1.0f / sqrtf(ss / (float) kEmbd + rms_eps);
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int i = k * kRowThreads + threadIdx.x;
        const float o = __fmul_rn(m[k] * r, norm_w[i]);
        y[(int64_t) t * kEmbd + i] = o;
        if (yq) {   // one warp per 32-block
            const float amax = q8_group_amax<32>(fabsf(o));
            yq[(int64_t) t * kEmbd + i] = q8_quant(o, q8_inv_scale(amax));
            if ((threadIdx.x & 31) == 0) yd[(int64_t) t * (kEmbd / 32) + i / 32] = q8_scale(amax);
        }
    }
}

__global__ void __launch_bounds__(kRowThreads)
rms_norm_kernel(const float * __restrict__ x, const float * __restrict__ add, const float * __restrict__ norm_w,
                float * __restrict__ y, float * __restrict__ sum_out, int n, int64_t ld, float eps) {
    __shared__ float red[kRowThreads / 32];
    const int64_t row = (int64_t) blockIdx.x * ld;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += kRowThreads) {
        float v = x[row + i];
        if (add) v = __fadd_rn(v, add[row + i]);
        if (sum_out) sum_out[row + i] = v;
        ss = fmaf(v, v, ss);
    }
    ss = block_sum(ss, red);
    const float r = 1.0f / sqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += kRowThreads) {
        float v = x[row + i];
        if (add) v = __fadd_rn(v, add[row + i]);
        y[row + i] = __fmul_rn(v * r, norm_w[i]);
    }
}

// One warp per token. NPL = experts per lane (n_expert / 32, rounded up).
template <int NPL>
__global__ void moe_route_kernel(const float * __restrict__ logits, const float * __restrict__ bias, int n_expert,
                                 int n_used, float floor_, float scale, int T, int32_t * __restrict__ ids,
                                 float * __restrict__ w) {
    const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (t >= T) return;
    const int lane = threadIdx.x & 31;
    float p[NPL], sel[NPL];
#pragma unroll
    for (int k = 0; k < NPL; ++k) {
        const int e = lane + 32 * k;
        if (e < n_expert) {
            p[k] = sigmoidf_(logits[(int64_t) t * n_expert + e]);
            sel[k] = __fadd_rn(p[k], bias[e]);
        } else {
            p[k] = 0.0f;
            sel[k] = -INFINITY;
        }
    }
    float wsel = 0.0f, sum = 0.0f;
    int my_id = -1;
    for (int j = 0; j < n_used; ++j) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int k = 0; k < NPL; ++k)
            if (sel[k] > bv) { bv = sel[k]; bi = lane + 32 * k; }   // ascending e: first max wins locally
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        float pw = 0.0f;
#pragma unroll
        for (int k = 0; k < NPL; ++k)
            if (lane + 32 * k == bi) { pw = p[k]; sel[k] = -INFINITY; }
        pw = __shfl_sync(0xffffffffu, pw, bi & 31);
        sum += pw;
        if (lane == j) { my_id = bi; wsel = pw; }
    }
    if (lane < n_used) {
        const float s = fmaxf(sum, floor_);
        ids[(int64_t) t * n_used + lane] = my_id;
        w[(int64_t) t * n_used + lane] = (wsel / s) * scale;
    }
}

__global__ void swiglu_clamp_kernel(const float * __restrict__ gate, const float * __restrict__ up,
                                    float * __restrict__ h, int8_t * __restrict__ hq, float * __restrict__ hd, int F,
                                    int64_t ld_in, int64_t ldh, float c, int64_t total) {
    // with hq, F % 32 == 0: every warp iteration covers exactly one 32-block of one row
    for (int64_t n = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; n < total; n += (int64_t) gridDim.x * blockDim.x) {
        const int64_t t = n / F;
        const int j = (int) (n % F);
        const float gv = fminf(gate[t * ld_in + j], c);
        const float uv = fmaxf(fminf(up[t * ld_in + j], c), -c);
        const float o = gv / (1.0f + expf(-gv)) * uv;
        h[t * ldh + j] = o;
        if (hq) {
            const float amax = q8_group_amax<32>(fabsf(o));
            hq[n] = q8_quant(o, q8_inv_scale(amax));
            if ((threadIdx.x & 31) == 0) hd[n / 32] = q8_scale(amax);
        }
    }
}

// One block of 1024 threads per row.
__global__ void __launch_bounds__(1024) argmax_kernel(const float * __restrict__ x, int n, int32_t * __restrict__ out) {
    const float * r = x + (int64_t) blockIdx.x * n;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int j = threadIdx.x; j < n; j += 1024)
        if (r[j] > bv) { bv = r[j]; bi = j; }
    __shared__ float sv[32];
    __shared__ int si[32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { sv[threadIdx.x / 32] = bv; si[threadIdx.x / 32] = bi; }
    __syncthreads();
    if (threadIdx.x < 32) {
        bv = sv[threadIdx.x];
        bi = si[threadIdx.x];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (threadIdx.x == 0) out[blockIdx.x] = bi;
    }
}

// T < kMixTileMin (decode, verify rows): one token per block row; else the token-tiled kernel.
template <int MODE, bool NEXT, int WT>
void launch_mix_t(int T, cudaStream_t st, const float * src, const float * src2, const float * hcp,
                  const float * x_in, float * x, const FnView & fn, float * part) {
    if (T >= kMixTileMin)
        hc_mix_tiled_kernel<MODE, NEXT, WT><<<dim3(kHcSplit, (T + kTileT - 1) / kTileT), kTileThreads, 0, st>>>(
            src, src2, hcp, x_in, x, fn, part, T);
    else
        hc_mix_kernel<MODE, NEXT, WT><<<dim3(kHcSplit, T), kMixThreads, 0, st>>>(src, src2, hcp, x_in, x, fn, part);
}

template <int MODE, bool NEXT>
void launch_mix(int T, cudaStream_t st, const float * src, const float * src2, const float * hcp,
                const float * x_in, float * x, const FnView & fn, float * part) {
    switch (fn.type) {
        case kTypeQ8_0R: launch_mix_t<MODE, NEXT, kTypeQ8_0R>(T, st, src, src2, hcp, x_in, x, fn, part); break;
        case kTypeQ8_0: launch_mix_t<MODE, NEXT, kTypeQ8_0>(T, st, src, src2, hcp, x_in, x, fn, part); break;
        case kTypeBF16: launch_mix_t<MODE, NEXT, kTypeBF16>(T, st, src, src2, hcp, x_in, x, fn, part); break;
        case kTypeF16: launch_mix_t<MODE, NEXT, kTypeF16>(T, st, src, src2, hcp, x_in, x, fn, part); break;
        default: launch_mix_t<MODE, NEXT, kTypeF32>(T, st, src, src2, hcp, x_in, x, fn, part); break;
    }
}

struct Q8Out {
    int8_t * qs = nullptr;
    float * d = nullptr;
};
Q8Out q8_out(void * p, int T, int n) {
    if (!p) return {};
    return {static_cast<int8_t *>(p), reinterpret_cast<float *>(static_cast<char *>(p) + q8_scales_offset(T, n))};
}

void launch_check(const char * what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

void hc_init(const Geometry & g, const float * e, float * x, int T, const HcBranch & next, float * part,
             cudaStream_t st) {
    check_geometry(g);
    if (T <= 0) return;
    const FnView fn = fn_view(next);
    launch_mix<0, true>(T, st, e, nullptr, nullptr, nullptr, x, fn, part);
    launch_check("hc_init");
}

void hc_post(const Geometry & g, const float * branch, const float * branch2, const float * hcp, const float * x_in,
             float * x_out, int T, const HcBranch * next, float * part, cudaStream_t st) {
    check_geometry(g);
    if (T <= 0) return;
    if (x_in == x_out) fail("hc_post: x_out must not alias x_in");
    if (next)
        launch_mix<1, true>(T, st, branch, branch2, hcp, x_in, x_out, fn_view(*next), part);
    else
        launch_mix_t<1, false, kTypeF32>(T, st, branch, branch2, hcp, x_in, x_out, FnView{}, nullptr);
    launch_check("hc_post");
}

void hc_pre(const Geometry & g, const HcBranch & br, const float * x, const float * part, int T,
            const WeightRef & norm_w, float * hcp, float * y, cudaStream_t st, void * y_q8) {
    check_geometry(g);
    if (T <= 0) return;
    check_f32(br.base, kMix, "hc_base");
    check_f32(br.scale, 3, "hc_scale");
    check_f32(norm_w, kEmbd, "branch norm");
    const Q8Out q = q8_out(y_q8, T, kEmbd);
    hc_pre_kernel<<<T, kRowThreads, 0, st>>>(x, part, static_cast<const float *>(br.base.data),
                                             static_cast<const float *>(br.scale.data),
                                             static_cast<const float *>(norm_w.data), hcp, y, q.qs, q.d, g.hc_eps,
                                             g.rms_eps);
    launch_check("hc_pre");
}

void hc_comb(const Geometry & g, float * hcp, int T, cudaStream_t st) {
    check_geometry(g);
    if (T <= 0) return;
    hc_comb_kernel<<<(T + 31) / 32, 32, 0, st>>>(hcp, T, g.hc_eps, g.hc_sinkhorn);
    launch_check("hc_comb");
}

void hc_head(const Geometry & g, const float * x, int T, const WeightRef & norm_w, float * y, cudaStream_t st,
             void * y_q8) {
    check_geometry(g);
    if (T <= 0) return;
    check_f32(norm_w, kEmbd, "output_norm");
    const Q8Out q = q8_out(y_q8, T, kEmbd);
    hc_head_kernel<<<T, kRowThreads, 0, st>>>(x, static_cast<const float *>(norm_w.data), y, q.qs, q.d, g.rms_eps);
    launch_check("hc_head");
}

void rms_norm(const float * x, const float * add, const WeightRef & norm_w, float * y, float * sum_out, int T, int n,
              int64_t ld, float eps, cudaStream_t st) {
    if (T <= 0) return;
    check_f32(norm_w, n, "rms_norm weight");
    rms_norm_kernel<<<T, kRowThreads, 0, st>>>(x, add, static_cast<const float *>(norm_w.data), y, sum_out, n,
                                               ld ? ld : n, eps);
    launch_check("rms_norm");
}

void moe_route(const Geometry & g, const float * logits, const WeightRef & bias, int T, int32_t * ids, float * w,
               cudaStream_t st) {
    if (T <= 0) return;
    check_f32(bias, g.n_expert, "exp_probs_b");
    if (g.n_expert > 9 * 32 || g.n_expert_used > 32) fail("router supports up to 288 experts, 32 used");
    constexpr int kWarps = 4;
    moe_route_kernel<9><<<(T + kWarps - 1) / kWarps, kWarps * 32, 0, st>>>(
        logits, static_cast<const float *>(bias.data), g.n_expert, g.n_expert_used, g.weight_norm_floor,
        g.expert_scale, T, ids, w);
    launch_check("moe_route");
}

void swiglu_clamp(const float * gate, const float * up, float * h, int T, int F, int64_t ld_in, int64_t ldh, float c,
                  cudaStream_t st, void * h_q8) {
    const int64_t total = (int64_t) T * F;
    if (total <= 0) return;
    if (h_q8 && F % 32) fail("swiglu_clamp: the Q8 output needs F % 32 == 0");
    const Q8Out q = q8_out(h_q8, T, F);
    const int blocks = (int) std::min<int64_t>((total + 255) / 256, 4096);
    swiglu_clamp_kernel<<<blocks, 256, 0, st>>>(gate, up, h, q.qs, q.d, F, ld_in, ldh, c, total);
    launch_check("swiglu_clamp");
}

void argmax_rows(const float * x, int n, int T, int32_t * out, cudaStream_t st) {
    if (T <= 0) return;
    argmax_kernel<<<T, 1024, 0, st>>>(x, n, out);
    launch_check("argmax_rows");
}

}  // namespace strata::glm
