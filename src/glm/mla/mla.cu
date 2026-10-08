// GLM-5.3-Flash MLA attention + DSA lightning indexer (see include/strata/glm/mla.hpp and
// docs/glm5next-spec.md section 3). Reference: llama.cpp src/models/glm5-next.cpp build_dsa_layer /
// build_kpool_select and src/llama-memory-hybrid-idx.cpp (pool grouping, visibility, tail cells).

#include "strata/glm/mla.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#include "strata/glm/matmul.hpp"
#include "strata/glm/q8_block.cuh"

namespace strata::glm {

namespace {

constexpr int kH = 64;          // attention heads
constexpr int kDq = 256;        // q/k/v head dim (no rope)
constexpr int kR = 512;         // latent rank
constexpr int kQr = 1536;       // q LoRA rank
constexpr int kIH = 32;         // indexer heads
constexpr int kID = 128;        // indexer head dim
constexpr int kPool = 4;        // cells per pool
constexpr int kTopPools = 512;  // idx_top_k / kpool
constexpr int kSelMax = kTopPools * kPool + kPool - 1;   // 2051 cells: selected pools + incomplete tail
constexpr int kSelLd = 2052;
constexpr int kChunk = 512;     // tokens per internal sub-chunk
constexpr int kSplitMaxT = 8;   // decode / verify batches (T <= 8) use the split attention kernel
constexpr int kFlashSplit1 = 64;    // cells per attention split at T = 1 (66 blocks at full selection)
constexpr int kFlashSplitN = 256;   // at 2 <= T <= kSplitMaxT (one wave of 72 blocks at T = 4)
constexpr int kMaxSplits = (kSelMax + kFlashSplit1 - 1) / kFlashSplit1;   // 33
constexpr int kPooledPad = 128; // pooled rows past the last pool, read (not used) by the tensor-core scorer
// key | gate rows on the device: a ring of the last kKgRing positions (pool_new reads at most the 3 rows before a
// sub-chunk plus the sub-chunk itself); every row also goes to a mapped host copy, read when the ring lost it
constexpr int kKgRing = 1024;
static_assert((kKgRing & (kKgRing - 1)) == 0 && kKgRing >= kChunk + kPool - 1, "kg ring: power of two, one sub-chunk + 3");

#define MLA_CUDA(x)                                                                                 \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "mla: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::abort();                                                                           \
        }                                                                                           \
    } while (0)

void check_geometry(const Geometry & g) {
    if (g.mla_heads != kH || g.mla_head_dim != kDq || g.mla_kv_rank != kR || g.mla_q_rank != kQr ||
        g.idx_heads != kIH || g.idx_head_dim != kID || g.idx_kpool != kPool || g.idx_top_k != kTopPools * kPool ||
        g.n_embd % 32 != 0) {
        std::fprintf(stderr, "mla: geometry differs from the compiled GLM-5.3-Flash MLA shapes\n");
        std::abort();
    }
}

int64_t pool_ld(int64_t max_ctx) { return (max_ctx / kPool + 3) / 4 * 4; }

// The decode / verify batches select in two stages once the context can span several slices of kSelSlice pools.
constexpr int kSelSlicePools = 4096;
int64_t cand_ld(int TC, int64_t max_ctx) {
    const int64_t n_slices = (max_ctx / kPool + kSelSlicePools - 1) / kSelSlicePools;
    return TC <= kSplitMaxT && n_slices > 1 ? n_slices * kTopPools : 0;
}

// Scratch layout of one sub-chunk of up to TC tokens.
struct Layout {
    size_t qr, q, qabs, o, kv, iq, ik, ig, iw, score, top, sel, nsel, qh, iqh, part_o, part_ml, ckey, cidx, mm_in, mm_w,
        mm_in2, total;
    size_t mm_in_bytes, mm_w_bytes, mm_in2_bytes;
    int64_t n_pool_ld;
    int64_t cand_ld;   // two-stage selection: candidates per token (0: single stage)
};

WeightRef shape_ref(int type, int64_t n_in, int64_t n_out, int64_t batch = 1) {
    WeightRef w;
    w.type = type;
    w.ne[0] = n_in;
    w.ne[1] = n_out;
    w.ne[2] = batch;
    return w;
}
constexpr int kQ8_0 = 8;

Layout layout(int TC, int64_t max_ctx, int n_embd) {
    Layout L{};
    // glm_matmul scratch: the largest prepared input (v_b: 64 heads x 512 per token) and the largest weight
    // dequantized for the GEMM path (attn_output); dense projections are Q8_0 (the quality floor of the port)
    L.mm_in_bytes = std::max({matmul_prepare_bytes(kR, TC, kH), matmul_prepare_bytes(kH * kDq, TC, 1),
                              matmul_prepare_bytes(n_embd, TC, 1)});
    L.mm_w_bytes = std::max({matmul_run_bytes(shape_ref(kQ8_0, kH * kDq, n_embd), TC),
                             matmul_run_bytes(shape_ref(kQ8_0, kQr, kH * kDq), TC),
                             matmul_run_bytes(shape_ref(kQ8_0, kDq, kR, kH), TC),
                             matmul_run_bytes(shape_ref(kQ8_0, kR, kDq, kH), TC),
                             matmul_run_bytes(shape_ref(kQ8_0, n_embd, kQr), TC),
                             matmul_run_bytes(shape_ref(kQ8_0, kQr, kIH * kID), TC)});
    // GEMV path: a second prepared input written by fused producers (q_resid by its RMSNorm, the per-head o by
    // attn_combine) while the side stream may still read mm_in
    L.mm_in2_bytes = TC <= kMatmulGemvMaxT
                         ? std::max(matmul_prepare_bytes(kQr, TC, 1), matmul_prepare_bytes(kR, TC, kH))
                         : 0;
    size_t off = 0;
    auto take = [&](size_t bytes) {
        const size_t at = off;
        off += (bytes + 255) / 256 * 256;
        return at;
    };
    L.n_pool_ld = pool_ld(max_ctx);
    L.qr = take(sizeof(float) * TC * kQr);
    L.q = take(sizeof(float) * TC * kH * kDq);
    L.qabs = take(sizeof(float) * TC * kH * kR);
    L.o = take(sizeof(float) * TC * kH * kR);
    L.kv = take(sizeof(float) * TC * kR);
    L.iq = take(sizeof(float) * TC * kIH * kID);
    L.ik = take(sizeof(float) * TC * kID);
    L.ig = take(sizeof(float) * TC * kID);
    L.iw = take(sizeof(float) * TC * kIH);
    L.score = take(sizeof(float) * TC * L.n_pool_ld);
    L.top = take(sizeof(int32_t) * TC * kTopPools);
    L.sel = take(sizeof(int32_t) * TC * kSelLd);
    L.nsel = take(sizeof(int32_t) * TC);
    L.qh = take(sizeof(__half) * TC * kH * kR);
    L.iqh = take(sizeof(__half) * TC * kIH * kID);
    const int TS = std::min(TC, kSplitMaxT);
    L.part_o = take(sizeof(float) * TS * kMaxSplits * kH * kR);
    L.part_ml = take(sizeof(float) * 2 * TS * kMaxSplits * kH);
    L.cand_ld = cand_ld(TC, max_ctx);
    L.ckey = take(sizeof(float) * TS * L.cand_ld);
    L.cidx = take(sizeof(int32_t) * TS * L.cand_ld);
    L.mm_in = take(L.mm_in_bytes);
    L.mm_w = take(L.mm_w_bytes);
    L.mm_in2 = take(L.mm_in2_bytes);
    L.total = off;
    return L;
}

// ---------------------------------------------------------------------------------------------------------
// Batch position: pos_off + *dpos when the engine passes it in device memory (captured decode step), else pos_off.
__device__ __forceinline__ int64_t batch_pos0(const int64_t * dpos, int64_t pos_off) {
    return dpos ? pos_off + *dpos : pos_off;
}

// ---------------------------------------------------------------------------------------------------------
// small row ops

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// block-wide sum for blockDim.x <= 1024; `red` holds 32 floats
__device__ float block_sum(float v, float * red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float r = lane < nw ? red[lane] : 0.f;
    return warp_sum(r);
}

// x[t] <- x[t] / sqrt(mean(x^2) + eps) * w
__global__ void rmsnorm_rows(float * x, int n, const float * __restrict__ w, float eps) {
    __shared__ float red[32];
    float * row = x + (int64_t)blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += row[i] * row[i];
    ss = block_sum(ss, red);
    const float scale = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) row[i] = row[i] * scale * w[i];
}

// rmsnorm_rows of N-element rows that also writes the row as glm_matmul's Q8 prepared input (q8_block.cuh layout,
// column blockIdx.x of gridDim.x). 256 threads, N % 256 == 0, so each warp step covers one aligned 32-element block;
// the row and the weights stay in registers between the two passes.
template <int N>
__global__ void __launch_bounds__(256) rmsnorm_rows_q8(float * x, const float * __restrict__ w, float eps,
                                                       void * prepared) {
    static_assert(N % 256 == 0, "one aligned Q8 block per warp step");
    constexpr int kPer = N / 256;
    __shared__ float red[32];
    float * row = x + (int64_t)blockIdx.x * N;
    float xv[kPer], wv[kPer];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        xv[j] = row[threadIdx.x + j * 256];
        wv[j] = w[threadIdx.x + j * 256];
        ss += xv[j] * xv[j];
    }
    ss = block_sum(ss, red);
    const float scale = rsqrtf(ss / N + eps);
    int8_t * qs = static_cast<int8_t *>(prepared) + (int64_t)blockIdx.x * N;
    float * qd = reinterpret_cast<float *>(static_cast<char *>(prepared) + q8_scales_offset(gridDim.x, N)) +
                 (int64_t)blockIdx.x * (N / 32);
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int i = threadIdx.x + j * 256;
        const float v = xv[j] * scale * wv[j];
        row[i] = v;
        const float amax = q8_group_amax<32>(fabsf(v));
        qs[i] = q8_quant(v, q8_inv_scale(amax));
        if ((threadIdx.x & 31) == 0) qd[i / 32] = q8_scale(amax);
    }
}

// x[t] <- (x[t] - mean) / sqrt(var + eps) * w + b   (n == blockDim.x)
__global__ void layernorm_rows(float * x, int n, const float * __restrict__ w, const float * __restrict__ b,
                               float eps) {
    __shared__ float red[32];
    float * row = x + (int64_t)blockIdx.x * n;
    const float v = row[threadIdx.x];
    const float mean = block_sum(v, red) / n;
    const float d = v - mean;
    const float var = block_sum(d * d, red) / n;
    row[threadIdx.x] = d * rsqrtf(var + eps) * w[threadIdx.x] + b[threadIdx.x];
}

__global__ void scale_rows(float * x, int64_t n, float s) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// The key | gate rows: ring slot pos % kKgRing (tagged with pos) and the host copy row pos
struct KgRows {
    __half * ring;          // [kKgRing][256] device
    int64_t * tag;          // [kKgRing] position held by each slot, -1 when none
    __half * host;          // [cap][256] mapped pinned host memory (device pointer)
};

__device__ __forceinline__ void kg_store(const KgRows & kg, int64_t pos, int i, __half key, __half gate) {
    const int64_t slot = pos & (kKgRing - 1);
    kg.ring[slot * 2 * kID + i] = key;
    kg.ring[slot * 2 * kID + kID + i] = gate;
    kg.host[pos * 2 * kID + i] = key;
    kg.host[pos * 2 * kID + kID + i] = gate;
    if (i == 0) kg.tag[slot] = pos;
}

// row q, from the ring when its slot still holds q (stream order: the writes come from an earlier kernel)
__device__ __forceinline__ const __half * kg_row(const KgRows & kg, int64_t q) {
    const int64_t slot = q & (kKgRing - 1);
    return kg.tag[slot] == q ? kg.ring + slot * 2 * kID : kg.host + q * 2 * kID;
}

// The latent rows: F16 [cap][512], or (Q8 caches) int8 [cap][512] with an F16 scale per kQ8Group dims [cap][8]
// (symmetric, scale = max |x| / 127 rounded to F16; 528 B per row instead of 1024)
constexpr int kQ8Group = 64;
struct LatentRows {
    __half * f16;      // null in a Q8 cache
    int8_t * q8;       // null in an F16 cache
    __half * scale;
};

// Row pos <- (a, b) = dims (threadIdx.x, threadIdx.x + 256); all 256 threads of the block call it (Q8: block-wide
// group maxima through `wm`, 16 floats of shared memory)
__device__ __forceinline__ void latent_store(const LatentRows & L, int64_t pos, float a, float b, float * wm) {
    const int i = threadIdx.x;
    if (!L.q8) {
        L.f16[pos * kR + i] = __float2half(a);
        L.f16[pos * kR + i + 256] = __float2half(b);
        return;
    }
    static_assert(kR == 512 && kQ8Group == 64, "two warps per group, groups 0-3 from a and 4-7 from b");
    const int warp = i >> 5;
    const float ma = warp_max(fabsf(a)), mb = warp_max(fabsf(b));
    __syncthreads();   // wm free (a caller's earlier use)
    if ((i & 31) == 0) {
        wm[warp] = ma;
        wm[8 + warp] = mb;
    }
    __syncthreads();
    const int g = warp >> 1;
    const __half sa = __float2half(fmaxf(wm[2 * g], wm[2 * g + 1]) / 127.f);
    const __half sb = __float2half(fmaxf(wm[8 + 2 * g], wm[8 + 2 * g + 1]) / 127.f);
    const float fa = __half2float(sa), fb = __half2float(sb);
    L.q8[pos * kR + i] = (int8_t)(fa > 0.f ? fminf(fmaxf(rintf(a / fa), -127.f), 127.f) : 0.f);
    L.q8[pos * kR + i + 256] = (int8_t)(fb > 0.f ? fminf(fmaxf(rintf(b / fb), -127.f), 127.f) : 0.f);
    if ((i & 63) == 0) {
        L.scale[pos * (kR / kQ8Group) + g] = sa;
        L.scale[pos * (kR / kQ8Group) + 4 + g] = sb;
    }
}

// dims [d0, d0 + 8) of row `cell` as 8 F16 (d0 a multiple of 8)
__device__ __forceinline__ uint4 latent_load8(const LatentRows & L, int64_t cell, int d0) {
    if (!L.q8) return __ldg(reinterpret_cast<const uint4 *>(L.f16 + cell * kR + d0));
    const uint2 q = __ldg(reinterpret_cast<const uint2 *>(L.q8 + cell * kR + d0));
    const __half s = __ldg(L.scale + cell * (kR / kQ8Group) + d0 / kQ8Group);
    const char4 lo = *reinterpret_cast<const char4 *>(&q.x), hi = *reinterpret_cast<const char4 *>(&q.y);
    uint4 r;
    __half2 * h = reinterpret_cast<__half2 *>(&r);
    const __half2 s2 = __half2half2(s);
    h[0] = __hmul2(__floats2half2_rn((float)lo.x, (float)lo.y), s2);
    h[1] = __hmul2(__floats2half2_rn((float)lo.z, (float)lo.w), s2);
    h[2] = __hmul2(__floats2half2_rn((float)hi.x, (float)hi.y), s2);
    h[3] = __hmul2(__floats2half2_rn((float)hi.z, (float)hi.w), s2);
    return r;
}

// dims (d, d + 1) of row `cell` (d even)
__device__ __forceinline__ float2 latent_load2(const LatentRows & L, int64_t cell, int d) {
    if (!L.q8) return __half22float2(*reinterpret_cast<const __half2 *>(L.f16 + cell * kR + d));
    const char2 q = *reinterpret_cast<const char2 *>(L.q8 + cell * kR + d);
    const float s = __half2float(__ldg(L.scale + cell * (kR / kQ8Group) + d / kQ8Group));
    return make_float2(__half2float(__float2half(q.x * s)), __half2float(__float2half(q.y * s)));
}

// latent[pos] = kv, idx_kg[pos] = key | gate (F16, like llama.cpp's cache rows); positions past cap are dropped.
// 256 threads per token.
__global__ void __launch_bounds__(256)
cache_write(const float * __restrict__ kv, const float * __restrict__ ik, const float * __restrict__ ig,
            const int64_t * __restrict__ dpos, int64_t pos_off, int64_t cap, LatentRows latent, KgRows kg) {
    static_assert(kR == 2 * 256 && kID <= 256, "two latent dims per thread");
    __shared__ float wm[16];
    const int t = blockIdx.x;
    const int64_t pos = batch_pos0(dpos, pos_off) + t;
    if (pos >= cap) return;
    latent_store(latent, pos, kv[t * kR + threadIdx.x], kv[t * kR + threadIdx.x + 256], wm);
    if (threadIdx.x < kID)
        kg_store(kg, pos, threadIdx.x, __float2half(ik[t * kID + threadIdx.x]), __float2half(ig[t * kID + threadIdx.x]));
}

// pooled[p][d] = sum_j softmax_j(gate[4p+j][d] + ape[j][d]) * key[4p+j][d] for the pools completed by T tokens at
// pos0: p in [pos0 / 4, (pos0 + T) / 4). Block b -> pool pos0 / 4 + b; the grid may hold more blocks than pools.
// rmsnorm_rows of kv, layernorm_rows of ik (same reductions, so the same bits) and cache_write in one launch of
// 256 threads per token
__global__ void __launch_bounds__(256)
norm_cache_write(float * __restrict__ kv, const float * __restrict__ kv_w, float rms_eps, float * __restrict__ ik,
                 const float * __restrict__ ik_w, const float * __restrict__ ik_b, float ln_eps,
                 const float * __restrict__ ig, const int64_t * __restrict__ dpos, int64_t pos_off, int64_t cap,
                 LatentRows latent, KgRows kg) {
    static_assert(kR == 2 * 256 && kID == 128, "two kv elements per thread, one ik element per thread of the first 128");
    __shared__ float red[32];
    const int t = blockIdx.x, i = threadIdx.x;
    const int64_t pos = batch_pos0(dpos, pos_off) + t;
    float * kr = kv + (int64_t)t * kR;
    const float a = kr[i], b = kr[i + 256];
    float ss = 0.f;
    ss += a * a;
    ss += b * b;
    ss = block_sum(ss, red);
    const float scale = rsqrtf(ss / kR + rms_eps);
    const float na = a * scale * kv_w[i], nb = b * scale * kv_w[i + 256];
    kr[i] = na;
    kr[i + 256] = nb;
    float * ir = ik + (int64_t)t * kID;
    const float v = i < kID ? ir[i] : 0.f;
    const float mean = block_sum(v, red) / kID;
    const float d = i < kID ? v - mean : 0.f;
    const float var = block_sum(d * d, red) / kID;
    const float nk = i < kID ? d * rsqrtf(var + ln_eps) * ik_w[i] + ik_b[i] : 0.f;
    if (i < kID) ir[i] = nk;
    if (pos >= cap) return;
    latent_store(latent, pos, na, nb, red);
    if (i < kID) kg_store(kg, pos, i, __float2half(nk), __float2half(ig[t * kID + i]));
}

__global__ void pool_new(const KgRows kg, const float * __restrict__ ape,
                         const int64_t * __restrict__ dpos, int64_t pos_off, int64_t T, int64_t cap,
                         __half * __restrict__ pooled) {
    const int64_t pos0 = batch_pos0(dpos, pos_off);
    const int64_t p = pos0 / kPool + blockIdx.x;
    if (p >= (pos0 + T) / kPool || (p + 1) * kPool > cap) return;
    const int d = threadIdx.x;
    float logit[kPool], key[kPool], m = -INFINITY;
#pragma unroll
    for (int j = 0; j < kPool; ++j) {
        const __half * row = kg_row(kg, p * kPool + j);
        key[j] = __half2float(row[d]);
        logit[j] = __half2float(row[kID + d]) + ape[j * kID + d];
        m = fmaxf(m, logit[j]);
    }
    float s = 0.f, acc = 0.f;
#pragma unroll
    for (int j = 0; j < kPool; ++j) {
        const float e = __expf(logit[j] - m);
        s += e;
        acc += e * key[j];
    }
    pooled[p * kID + d] = __float2half(acc / s);
}

// ---------------------------------------------------------------------------------------------------------
// indexer scores: score[t][p] = sum_h w[t][h] * relu(iq[t][h] . pooled[p]) for the pools visible to token t
// (p < (pos_t + 1) / 4). Lane = indexer head; the warp walks pools.

constexpr int kScoreWarps = 8;
constexpr int kScorePoolsPerBlock = 256;

__global__ void __launch_bounds__(kScoreWarps * 32)
score_pools(const float * __restrict__ iq, const float * __restrict__ iw, const __half * __restrict__ pooled,
            int64_t pos0, int64_t ld, float * __restrict__ score) {
    __shared__ float qT[kID][kIH];
    __shared__ float pk[kScoreWarps][kID];
    const int t = blockIdx.y;
    const int64_t nv = (pos0 + t + 1) / kPool;
    const int64_t pbeg = (int64_t)blockIdx.x * kScorePoolsPerBlock;
    if (pbeg >= nv) return;
    const int64_t pend = min(nv, pbeg + kScorePoolsPerBlock);
    for (int i = threadIdx.x; i < kIH * kID; i += blockDim.x) {
        const int h = i / kID, k = i % kID;
        qT[k][h] = iq[(int64_t)t * kIH * kID + i];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float w = iw[t * kIH + lane];
    for (int64_t p = pbeg + warp; p < pend; p += kScoreWarps) {
        const __half2 * src = reinterpret_cast<const __half2 *>(pooled + p * kID) + lane * 2;
        const float2 a = __half22float2(src[0]), b = __half22float2(src[1]);
        pk[warp][lane * 4 + 0] = a.x;
        pk[warp][lane * 4 + 1] = a.y;
        pk[warp][lane * 4 + 2] = b.x;
        pk[warp][lane * 4 + 3] = b.y;
        __syncwarp();
        float dot = 0.f;
#pragma unroll 16
        for (int k = 0; k < kID; ++k) dot += qT[k][lane] * pk[warp][k];
        __syncwarp();
        const float s = warp_sum(fmaxf(dot, 0.f) * w);
        if (lane == 0) score[(int64_t)t * ld + p] = s;
    }
}

// ---------------------------------------------------------------------------------------------------------
// selection: top min(nv, 512) pools by score, sorted descending, expanded to their cells, plus the incomplete
// tail (positions pos, pos-1, ... of the current pool). One block per token.

constexpr int kSelThreads = 512;

__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void __launch_bounds__(kSelThreads)
select_cells(const float * __restrict__ score, int64_t ld, int64_t pos0, int32_t * __restrict__ top,
             int32_t * __restrict__ sel, int32_t * __restrict__ nsel) {
    __shared__ uint32_t hist[256];
    __shared__ float key[kTopPools];
    __shared__ int32_t idx[kTopPools];
    __shared__ uint32_t s_prefix, s_k;
    __shared__ int s_n_gt, s_n_eq;
    const int t = blockIdx.x;
    const int64_t pos = pos0 + t;
    const int64_t nv = (pos + 1) / kPool;
    const float * row = score + (int64_t)t * ld;
    const int ntop = (int)(nv < kTopPools ? nv : kTopPools);

    if (nv <= kTopPools) {
        for (int i = threadIdx.x; i < kTopPools; i += blockDim.x) {
            key[i] = i < nv ? row[i] : -INFINITY;
            idx[i] = i < nv ? i : INT32_MAX;
        }
    } else {
        // radix select of the 512th largest key, 8 bits per pass from the top
        uint32_t prefix = 0, mask = 0, k = kTopPools;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
                const uint32_t u = order_key(row[i]);
                if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 0xff], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                uint32_t cum = 0;
                int d = 255;
                for (; d > 0; --d) {
                    if (cum + hist[d] >= k) break;
                    cum += hist[d];
                }
                s_prefix = prefix | ((uint32_t)d << shift);
                s_k = k - cum;
            }
            __syncthreads();
            prefix = s_prefix;
            k = s_k;
            mask |= 0xffu << shift;
            __syncthreads();
        }
        // prefix = threshold key; take every key above it and the first k equal to it (lowest pool ids)
        if (threadIdx.x == 0) {
            s_n_gt = 0;
            s_n_eq = 0;
        }
        __syncthreads();
        const int n_gt_total = kTopPools - (int)k;
        for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
            const uint32_t u = order_key(row[i]);
            if (u > prefix) {
                const int at = atomicAdd(&s_n_gt, 1);
                key[at] = row[i];
                idx[at] = (int32_t)i;
            }
        }
        __syncthreads();
        // exact float ties at the threshold: any k of them (llama.cpp's top_k does not order ties either)
        for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
            if (order_key(row[i]) == prefix) {
                const int at = n_gt_total + atomicAdd(&s_n_eq, 1);
                if (at < kTopPools) {
                    key[at] = row[i];
                    idx[at] = (int32_t)i;
                }
            }
        }
    }
    __syncthreads();
    // bitonic sort of 512 (key desc, then pool id asc)
    for (int size = 2; size <= kTopPools; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < kTopPools; i += blockDim.x) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool desc = (i & size) == 0;
                    const float ki = key[i], kj = key[j];
                    const int32_t ii = idx[i], ij = idx[j];
                    const bool i_first = ki > kj || (ki == kj && ii < ij);
                    if (i_first != desc) {
                        key[i] = kj;
                        key[j] = ki;
                        idx[i] = ij;
                        idx[j] = ii;
                    }
                }
            }
            __syncthreads();
        }
    }
    int32_t * trow = top + (int64_t)t * kTopPools;
    int32_t * srow = sel + (int64_t)t * kSelLd;
    for (int i = threadIdx.x; i < kTopPools; i += blockDim.x) {
        const int32_t p = i < ntop ? idx[i] : -1;
        trow[i] = p;
        if (i < ntop) {
#pragma unroll
            for (int j = 0; j < kPool; ++j) srow[i * kPool + j] = p * kPool + j;
        }
    }
    const int n_tail = (int)((pos + 1) % kPool);
    if (threadIdx.x < n_tail) srow[ntop * kPool + threadIdx.x] = (int32_t)(pos - threadIdx.x);
    if (threadIdx.x == 0) nsel[t] = ntop * kPool + n_tail;
}

// ---------------------------------------------------------------------------------------------------------
// sparse latent attention: o[t][h] = softmax_c(q_abs[t][h] . latent[c] / 16) latent[c] over the selected cells.
// One block per (token, group of kHG heads).

constexpr int kHG = 4;
constexpr int kAttnThreads = 256;

__global__ void __launch_bounds__(kAttnThreads)
attn_sparse(const float * __restrict__ qabs, const LatentRows latent, const int32_t * __restrict__ sel,
            const int32_t * __restrict__ nsel, int sel_ld, float scale, float * __restrict__ o) {
    __shared__ float q[kHG][kR];
    __shared__ float s[kHG][kSelMax];
    __shared__ float inv[kHG];
    const int t = blockIdx.x, h0 = blockIdx.y * kHG;
    const int n = nsel[t];
    const int32_t * cells = sel + (int64_t)t * sel_ld;
    for (int i = threadIdx.x; i < kHG * kR; i += blockDim.x)
        q[i / kR][i % kR] = qabs[((int64_t)t * kH + h0) * kR + i] * scale;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int c = warp; c < n; c += kAttnThreads / 32) {
        const int64_t cell = cells[c];
        float acc[kHG] = {};
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int d0 = part * 256 + lane * 8;
            const uint4 raw = latent_load8(latent, cell, d0);
            const __half2 * hv = reinterpret_cast<const __half2 *>(&raw);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 kv = __half22float2(hv[e]);
#pragma unroll
                for (int h = 0; h < kHG; ++h) acc[h] += q[h][d0 + 2 * e] * kv.x + q[h][d0 + 2 * e + 1] * kv.y;
            }
        }
#pragma unroll
        for (int h = 0; h < kHG; ++h) {
            const float v = warp_sum(acc[h]);
            if (lane == 0) s[h][c] = v;
        }
    }
    __syncthreads();
    if (warp < kHG) {
        float m = -INFINITY;
        for (int c = lane; c < n; c += 32) m = fmaxf(m, s[warp][c]);
        m = warp_max(m);
        float sum = 0.f;
        for (int c = lane; c < n; c += 32) {
            const float e = __expf(s[warp][c] - m);
            s[warp][c] = e;
            sum += e;
        }
        sum = warp_sum(sum);
        if (lane == 0) inv[warp] = 1.f / sum;
    }
    __syncthreads();
    // each thread owns two latent dims for all kHG heads
    const int d = threadIdx.x * 2;
    float acc[kHG][2] = {};
    for (int c = 0; c < n; ++c) {
        const float2 kv = latent_load2(latent, cells[c], d);
#pragma unroll
        for (int h = 0; h < kHG; ++h) {
            acc[h][0] += s[h][c] * kv.x;
            acc[h][1] += s[h][c] * kv.y;
        }
    }
#pragma unroll
    for (int h = 0; h < kHG; ++h) {
        float * dst = o + ((int64_t)t * kH + h0 + h) * kR + d;
        dst[0] = acc[h][0] * inv[h];
        dst[1] = acc[h][1] * inv[h];
    }
}

// =========================================================================================================
// Tensor-core (WMMA, sm_70) kernels. The v1 kernels above stay as the reference path (MLA_NAIVE=1).

using namespace nvcuda;

// F32 -> F16 rows with a scale (q_abs prescaled by 1/sqrt(256) = 1/16, exact in F16; indexer q unscaled)
__global__ void to_half_scaled(const float * __restrict__ x, int64_t n, float scale, __half * __restrict__ y) {
    const int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    const float4 v = *reinterpret_cast<const float4 *>(x + i);
    __half2 * dst = reinterpret_cast<__half2 *>(y + i);
    dst[0] = __floats2half2_rn(v.x * scale, v.y * scale);
    dst[1] = __floats2half2_rn(v.z * scale, v.w * scale);
}

// ---- indexer scores ---------------------------------------------------------------------------------------
// Block = 128 pools (8 warps x 16) x a group of tokens. Per warp the 16 pooled keys stay in A fragments; per token
// S[16 pools][32 heads] = pooled . iq^T on tensor cores, then relu * w summed over heads.
constexpr int kScTile = 128;     // pools per block
constexpr int kScTokens = 16;    // tokens per block
constexpr int kIqLd = kID + 8;

__global__ void __launch_bounds__(256)
score_wmma(const __half * __restrict__ iqh, const float * __restrict__ iw, const __half * __restrict__ pooled,
           const int64_t * __restrict__ dpos, int64_t pos_off, int T, int64_t ld, float * __restrict__ score) {
#if __CUDA_ARCH__ != 700
    __shared__ __align__(32) float sm[8][16][36];
#endif
    const int64_t pos0 = batch_pos0(dpos, pos_off);
    __shared__ __align__(32) __half qs[kIH][kIqLd];
    __shared__ float ws[kScTokens][kIH];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t pbeg = (int64_t)blockIdx.x * kScTile;
    const int t0 = blockIdx.y * kScTokens, tn = min(kScTokens, T - t0);
    // tokens see pools < (pos+1)/4; the last token of the group sees the most
    if (pbeg >= (pos0 + t0 + tn) / kPool) return;
    for (int i = threadIdx.x; i < tn * kIH; i += blockDim.x) ws[i / kIH][i % kIH] = iw[(int64_t)(t0 + i / kIH) * kIH + i % kIH];
    const int64_t p0 = pbeg + warp * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[kID / 16];
#pragma unroll
    for (int k = 0; k < kID / 16; ++k) wmma::load_matrix_sync(a[k], pooled + p0 * kID + k * 16, kID);
    for (int tt = 0; tt < tn; ++tt) {
        const int t = t0 + tt;
        const int64_t nv = (pos0 + t + 1) / kPool;
        if (pbeg >= nv) continue;   // block-uniform
        __syncthreads();
        {   // stage this token's 32 x 128 indexer queries: 512 uint4, 2 per thread
            const uint4 * src = reinterpret_cast<const uint4 *>(iqh + (int64_t)t * kIH * kID);
            const uint4 v0 = src[threadIdx.x], v1 = src[threadIdx.x + 256];
            *reinterpret_cast<uint4 *>(&qs[threadIdx.x / 16][(threadIdx.x % 16) * 8]) = v0;
            *reinterpret_cast<uint4 *>(&qs[(threadIdx.x + 256) / 16][(threadIdx.x % 16) * 8]) = v1;
        }
        __syncthreads();
        if (p0 >= nv) continue;     // warp-uniform
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
        wmma::fill_fragment(acc0, 0.f);
        wmma::fill_fragment(acc1, 0.f);
#pragma unroll
        for (int k = 0; k < kID / 16; ++k) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
            wmma::load_matrix_sync(b, &qs[0][k * 16], kIqLd);       // heads 0..15
            wmma::mma_sync(acc0, a[k], b, acc0);
            wmma::load_matrix_sync(b, &qs[16][k * 16], kIqLd);      // heads 16..31
            wmma::mma_sync(acc1, a[k], b, acc1);
        }
#if __CUDA_ARCH__ == 700
        // sm_70 accumulator layout: lane l holds rows R, R+2 (R = (l&1) + 4*((l>>4)&1) + 8*((l>>2)&1)) at columns
        // c0, c0+1, c0+4, c0+5 (c0 = 2*((l>>1)&1) + 8*((l>>3)&1)); x[0,1,4,5] are row R, x[2,3,6,7] row R+2. The four
        // lanes sharing a row differ in bits 1 and 3.
        const int c0 = ((lane >> 1) & 1) * 2 + ((lane >> 3) & 1) * 8;
        const int r = (lane & 1) + ((lane >> 4) & 1) * 4 + ((lane >> 2) & 1) * 8;
        const float * w = ws[tt];
        float v0 = 0.f, v1 = 0.f;
#pragma unroll
        for (int f = 0; f < 2; ++f) {
            const auto & acc = f ? acc1 : acc0;
            const float * wf = w + f * 16 + c0;
            v0 += fmaxf(acc.x[0], 0.f) * wf[0] + fmaxf(acc.x[1], 0.f) * wf[1] + fmaxf(acc.x[4], 0.f) * wf[4] +
                  fmaxf(acc.x[5], 0.f) * wf[5];
            v1 += fmaxf(acc.x[2], 0.f) * wf[0] + fmaxf(acc.x[3], 0.f) * wf[1] + fmaxf(acc.x[6], 0.f) * wf[4] +
                  fmaxf(acc.x[7], 0.f) * wf[5];
        }
        v0 += __shfl_xor_sync(0xffffffffu, v0, 2);
        v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
        v0 += __shfl_xor_sync(0xffffffffu, v0, 8);
        v1 += __shfl_xor_sync(0xffffffffu, v1, 8);
        if ((lane & 10) == 0) {
            if (p0 + r < nv) score[(int64_t)t * ld + p0 + r] = v0;
            if (p0 + r + 2 < nv) score[(int64_t)t * ld + p0 + r + 2] = v1;
        }
#else
        wmma::store_matrix_sync(&sm[warp][0][0], acc0, 36, wmma::mem_row_major);
        wmma::store_matrix_sync(&sm[warp][0][16], acc1, 36, wmma::mem_row_major);
        __syncwarp();
        const int r = lane >> 1, h0 = (lane & 1) * 16;
        float v = 0.f;
#pragma unroll
        for (int h = 0; h < 16; ++h) v += fmaxf(sm[warp][r][h0 + h], 0.f) * ws[tt][h0 + h];
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        if ((lane & 1) == 0 && p0 + r < nv) score[(int64_t)t * ld + p0 + r] = v;
        __syncwarp();
#endif
    }
}

// ---- top-k selection --------------------------------------------------------------------------------------
// One block of 512 threads per token. nv <= 512: every visible pool, sorted by descending score (the order
// llama.cpp reports). Otherwise the 512 best pools, in no particular order (the order does not enter the math):
//  1. the order keys sit in registers, 8 per thread per window of 4096 (larger rows re-read the scores per pass);
//  2. tau0 = min over threads of the thread maximum is a lower bound of the 512th largest key (the 512 thread
//     maxima are 512 distinct pools), so only keys >= tau0 are candidates (at most 4096), compacted to smem;
//  3. the 512th largest key is found by a histogram of value-linear bins over [tau0, max] and a one-warp rank of
//     the few candidates in the bin holding it (an 8-bit radix search when that bin is crowded), then keys above it
//     and enough ties are taken.
// If the candidates overflow, a bitwise search runs over all pools (slow, exact). Every placement is by ballot
// counts, so the result is deterministic.
constexpr int kSel2Threads = 512;
constexpr int kSelVpt = 8;       // 512 x 8 = 4096 keys in registers per window (unrolled loops stay small: the
                                 // V100 instruction cache thrashed at 64 keys per thread)
constexpr int kSelWin = kSel2Threads * kSelVpt;
constexpr int kSelCap = 4096;    // candidate slots in shared memory
constexpr int kSelCpt = kSelCap / kSel2Threads;
constexpr int kSelBins = 2 * kSel2Threads;   // value-linear bins of the first search round (2 per thread scan)
constexpr int kSelRank = 32;                 // a cut bin of at most this many keys is ranked by one warp

__device__ __forceinline__ float key_to_float(uint32_t u) {
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u);
}

// block-wide sum of per-thread counts (512 threads = 16 warps); `red` holds 2 x 16 slots, alternated by `parity`
__device__ __forceinline__ uint32_t block_count(uint32_t c, uint32_t * red, int parity) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    uint32_t * buf = red + parity * 16;
    if ((threadIdx.x & 31) == 0) buf[threadIdx.x >> 5] = c;
    __syncthreads();
    uint32_t total = 0;
#pragma unroll
    for (int w = 0; w < kSel2Threads / 32; ++w) total += buf[w];
    return total;
}

// exclusive prefix over the 16 warps of per-warp totals (lane 0 of each warp passes its total); all threads get
// their warp's offset and the block total
__device__ __forceinline__ uint32_t warp_offsets(uint32_t warp_total, uint32_t * red, uint32_t & block_total) {
    const int warp = threadIdx.x >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[warp] = warp_total;
    __syncthreads();
    uint32_t off = 0, tot = 0;
#pragma unroll
    for (int w = 0; w < kSel2Threads / 32; ++w) {
        const uint32_t v = red[w];
        off += w < warp ? v : 0u;
        tot += v;
    }
    block_total = tot;
    return off;
}

// Top min(nv, 512) of row[0, nv) into key / idx (shared, 512 each; idx = position in row). nv <= 512: every key,
// sorted by descending key then position when sort_small. Ends with the results visible to the whole block only
// after the caller's __syncthreads().
__device__ void top_pools(const float * __restrict__ row, int64_t nv, bool sort_small, float * key, int32_t * idx) {
    __shared__ uint32_t red[32];
    __shared__ uint32_t cand_u[kSelCap];
    __shared__ int32_t cand_i[kSelCap];
    __shared__ uint32_t s_digit, s_k;
    __shared__ uint32_t hist[2][256];
    __shared__ __align__(8) uint32_t vbin[kSelBins];
    __shared__ uint32_t s_bin[kSelRank];
    __shared__ uint32_t s_nbin, s_bmin, s_bmax;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (nv <= kTopPools) {
        for (int i = threadIdx.x; i < kTopPools; i += blockDim.x) {
            key[i] = i < nv ? __ldcg(row + i) : -INFINITY;
            idx[i] = i < nv ? i : INT32_MAX;
        }
        __syncthreads();
        if (!sort_small) return;
        // bitonic sort of 512 (key desc, then pool id asc), one compare-exchange per thread of the first 256
        for (int size = 2; size <= kTopPools; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < kTopPools / 2) {
                    const int i = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
                    const int j = i + stride;
                    const bool desc = (i & size) == 0;
                    const float ki = key[i], kj = key[j];
                    const int32_t ii = idx[i], ij = idx[j];
                    const bool i_first = ki > kj || (ki == kj && ii < ij);
                    if (i_first != desc) {
                        key[i] = kj;
                        key[j] = ki;
                        idx[i] = ij;
                        idx[j] = ii;
                    }
                }
                __syncthreads();
            }
        }
    } else {
        // 1. keys in register windows of 4096 (8 per thread; 0 past nv: below every real key); one window
        //    stays loaded, larger nv re-reads the scores per pass
        const int n_win = (int)((nv + kSelWin - 1) / kSelWin);
        uint32_t u[kSelVpt];
        auto load_win = [&](int wi) {
#pragma unroll
            for (int j = 0; j < kSelVpt; ++j) {
                const int64_t i = (int64_t)wi * kSelWin + threadIdx.x + (int64_t)j * kSel2Threads;
                const float f = __ldcg(row + (i < nv ? i : nv - 1));   // unconditional: the loads pipeline
                u[j] = i < nv ? order_key(f) : 0u;
            }
        };
        uint32_t tmax = 0;
        for (int wi = 0; wi < n_win; ++wi) {
            load_win(wi);
#pragma unroll
            for (int j = 0; j < kSelVpt; ++j) tmax = max(tmax, u[j]);
        }
        // 2. tau0 = block min of thread maxima, gmax = block max (one reduction)
        uint32_t wtau = tmax, wmax = tmax;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            wtau = min(wtau, __shfl_xor_sync(0xffffffffu, wtau, o));
            wmax = max(wmax, __shfl_xor_sync(0xffffffffu, wmax, o));
        }
        if (lane == 0) {
            red[warp] = wtau;
            red[16 + warp] = wmax;
        }
        if (threadIdx.x < 256) hist[0][threadIdx.x] = 0;
        vbin[threadIdx.x] = 0;
        vbin[threadIdx.x + kSel2Threads] = 0;
        if (threadIdx.x == 0) {
            s_nbin = 0;
            s_bmin = ~0u;
            s_bmax = 0;
        }
        __syncthreads();
        uint32_t tau0 = red[0], gmax = red[16];
#pragma unroll
        for (int w = 1; w < kSel2Threads / 32; ++w) {
            tau0 = min(tau0, red[w]);
            gmax = max(gmax, red[16 + w]);
        }
        // a single window is searched in place in the registers; longer rows compact the keys >= tau0 to shared
        // memory first
        const bool direct = n_win == 1;
        uint32_t n_c = (uint32_t)nv;
        if (!direct) {
            uint32_t wcnt = 0;   // ballots are warp-uniform: count, then place without atomics
            for (int wi = 0; wi < n_win; ++wi) {
                load_win(wi);
#pragma unroll
                for (int j = 0; j < kSelVpt; ++j) wcnt += __popc(__ballot_sync(0xffffffffu, u[j] >= tau0));
            }
            uint32_t total = 0;
            uint32_t at = warp_offsets(wcnt, red, total);
            n_c = total;
            if (n_c <= (uint32_t)kSelCap) {
                const uint32_t below = (1u << lane) - 1;
                for (int wi = 0; wi < n_win; ++wi) {
                    if (n_win > 1) load_win(wi);
#pragma unroll
                    for (int j = 0; j < kSelVpt; ++j) {
                        const bool p = u[j] >= tau0;
                        const uint32_t m = __ballot_sync(0xffffffffu, p);
                        if (p) {
                            cand_u[at + __popc(m & below)] = u[j];
                            cand_i[at + __popc(m & below)] = (int32_t)(wi * kSelWin + threadIdx.x + j * kSel2Threads);
                        }
                        at += __popc(m);
                    }
                }
                __syncthreads();
            }
        }
        const bool fast = n_c <= (uint32_t)kSelCap;
        // keys below tau0 cannot be selected: they are left out of the search (fewer histogram atomics, and the
        // search starts below the bits tau0 and the largest key share, past the sign / exponent bits)
        const uint32_t lo_key = tau0;
        if (fast) {
            // 3. radix search over the candidates
            static_assert(kSelCpt == kSelVpt, "a direct window is laid out like the candidates");
            uint32_t cu[kSelCpt];
#pragma unroll
            for (int j = 0; j < kSelCpt; ++j) {
                const uint32_t i = threadIdx.x + j * kSel2Threads;
                cu[j] = direct ? u[j] : i < n_c ? cand_u[i] : 0u;
            }
            // the 512th largest key lies in [lo_key, hi_key]; kk = its rank from the top among the candidates there
            uint32_t lo_k = lo_key, hi_key = gmax, kk = kTopPools;
            uint32_t prefix = 0;
            bool resolved = false;
            // value-linear bins over [tau0, gmax] (scores bunch in a few exponents, so key-bit digits would pile
            // the histogram atomics onto a few bins): the bin holding the 512th key keeps a few candidates, ranked
            // by one warp. Bins come from one rounded expression (monotone in the key), kept per candidate.
            const float f_lo = key_to_float(lo_key), f_rng = key_to_float(gmax) - f_lo;
            const float b_scale = (float)kSelBins / f_rng;
            if (gmax > lo_key && isfinite(f_rng) && isfinite(b_scale) && b_scale > 0.f) {
                uint32_t cb[kSelCpt];
#pragma unroll
                for (int j = 0; j < kSelCpt; ++j) {
                    const uint32_t i = threadIdx.x + j * kSel2Threads;
                    const float f = key_to_float(cu[j]);
                    const float x = __fmul_rn(__fsub_rn(f, f_lo), b_scale);
                    cb[j] = f != f || x >= (float)(kSelBins - 1) ? kSelBins - 1 : x > 0.f ? (uint32_t)x : 0u;
                    if (i < n_c && cu[j] >= lo_key) atomicAdd(&vbin[cb[j]], 1u);
                }
                __syncthreads();
                // thread i scans bins 1023 - 2i and 1022 - 2i (largest first), block prefix over threads
                const uint2 c2 = *reinterpret_cast<const uint2 *>(&vbin[kSelBins - 2 - 2 * threadIdx.x]);
                const uint32_t c_hi = c2.y, s = c2.x + c2.y;
                uint32_t incl = s;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const uint32_t v = __shfl_up_sync(0xffffffffu, incl, o);
                    if (lane >= o) incl += v;
                }
                uint32_t wtot = __shfl_sync(0xffffffffu, incl, 31), btot = 0;
                const uint32_t excl = warp_offsets(wtot, red, btot) + incl - s;
                if (excl < kk && kk <= excl + s) {
                    const bool upper = kk <= excl + c_hi;
                    s_digit = kSelBins - 1 - 2 * threadIdx.x - (upper ? 0 : 1);
                    s_k = kk - excl - (upper ? 0 : c_hi);
                }
                __syncthreads();
                const uint32_t cut = s_digit;
                kk = s_k;
                // candidates of the cut bin: the first kSelRank to shared memory, plus their key range
#pragma unroll
                for (int j = 0; j < kSelCpt; ++j) {
                    const uint32_t i = threadIdx.x + j * kSel2Threads;
                    if (i < n_c && cu[j] >= lo_key && cb[j] == cut) {
                        const uint32_t slot = atomicAdd(&s_nbin, 1u);
                        if (slot < (uint32_t)kSelRank) s_bin[slot] = cu[j];
                        atomicMin(&s_bmin, cu[j]);
                        atomicMax(&s_bmax, cu[j]);
                    }
                }
                __syncthreads();
                const uint32_t m = s_nbin;
                if (m <= (uint32_t)kSelRank) {
                    // rank of each key in the bin: the kk-th largest is the key with (greater) < kk <= (greater + equal)
                    if (warp == 0) {
                        const uint32_t v = lane < (int)m ? s_bin[lane] : 0u;
                        uint32_t gt = 0, eq = 0;
                        for (uint32_t j = 0; j < m; ++j) {
                            const uint32_t w = s_bin[j];
                            gt += w > v;
                            eq += w == v;
                        }
                        if (lane < (int)m && gt < kk && kk <= gt + eq) s_k = v;
                    }
                    __syncthreads();
                    prefix = s_k;
                    resolved = true;
                } else {
                    lo_k = s_bmin;
                    hi_key = s_bmax;
                }
            }
            // otherwise radix search of the kk-th largest key in [lo_k, hi_key], 8 bits per round from the first bit
            // where the bounds differ (all keys there share the bits above it): a shared histogram of the keys
            // matching the prefix so far, scanned by one warp
            const int top_bit = (lo_k ^ hi_key) ? 31 - __clz(lo_k ^ hi_key) : -1;
            if (!resolved) prefix = top_bit >= 31 ? 0u : (lo_k >> (top_bit + 1)) << (top_bit + 1);
            int hi = resolved ? 0 : top_bit + 1;   // bits [hi, 32) of the answer are known
            int hb = 0;
            while (hi > 0) {
                const int lo = hi > 8 ? hi - 8 : 0;
                const uint32_t hmask = hi >= 32 ? 0u : ~0u << hi, dmask = (1u << (hi - lo)) - 1;
                uint32_t * h = hist[hb];
#pragma unroll
                for (int j = 0; j < kSelCpt; ++j) {
                    const uint32_t i = threadIdx.x + j * kSel2Threads;
                    if (i < n_c && cu[j] >= lo_k && cu[j] <= hi_key && (cu[j] & hmask) == (prefix & hmask))
                        atomicAdd(&h[(cu[j] >> lo) & dmask], 1u);
                }
                if (threadIdx.x < 256) hist[hb ^ 1][threadIdx.x] = 0;   // for the next round
                __syncthreads();
                if (threadIdx.x < 32) {
                    // lane l holds bins [8 (31 - l), 8 (31 - l) + 8): lane 0 the largest digits
                    const int base = (31 - lane) * 8;
                    uint32_t c[8], sum = 0;
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        c[e] = h[base + e];
                        sum += c[e];
                    }
                    uint32_t incl = sum;
#pragma unroll
                    for (int o = 1; o < 32; o <<= 1) {
                        const uint32_t v = __shfl_up_sync(0xffffffffu, incl, o);
                        if (lane >= o) incl += v;
                    }
                    uint32_t acc = incl - sum;   // candidates in larger digits than this lane's bins
                    if (acc < kk && kk <= incl) {
#pragma unroll
                        for (int e = 7; e >= 0; --e) {
                            if (acc < kk && kk <= acc + c[e]) {
                                s_digit = (uint32_t)(base + e);
                                s_k = kk - acc;
                            }
                            acc += c[e];
                        }
                    }
                }
                __syncthreads();
                prefix |= s_digit << lo;
                kk = s_k;
                hi = lo;
                hb ^= 1;
            }
            // take keys above prefix, then ties, placed by per-warp counts (no atomics); the two counts (<= 4096)
            // share one offset scan, packed in 16-bit halves
            uint32_t wcnt2 = 0;
            const uint32_t below = (1u << lane) - 1;
#pragma unroll
            for (int j = 0; j < kSelCpt; ++j) {
                const bool v = threadIdx.x + j * kSel2Threads < n_c;
                wcnt2 += __popc(__ballot_sync(0xffffffffu, v && cu[j] > prefix)) +
                         (__popc(__ballot_sync(0xffffffffu, v && cu[j] == prefix)) << 16);
            }
            uint32_t tot2 = 0;
            const uint32_t off2 = warp_offsets(wcnt2, red, tot2);
            const uint32_t n_gt = tot2 & 0xffffu;
            uint32_t at_gt = off2 & 0xffffu, at_eq = n_gt + (off2 >> 16);
#pragma unroll
            for (int j = 0; j < kSelCpt; ++j) {
                const uint32_t i = threadIdx.x + j * kSel2Threads;
                const bool v = i < n_c;
                const bool pg = v && cu[j] > prefix, pe = v && cu[j] == prefix;
                const uint32_t mg = __ballot_sync(0xffffffffu, pg), me = __ballot_sync(0xffffffffu, pe);
                const uint32_t sg = at_gt + __popc(mg & below), se = at_eq + __popc(me & below);
                if (pg) {
                    key[sg] = key_to_float(cu[j]);
                    idx[sg] = direct ? (int32_t)i : cand_i[i];
                }
                if (pe && se < (uint32_t)kTopPools) {
                    key[se] = key_to_float(cu[j]);
                    idx[se] = direct ? (int32_t)i : cand_i[i];
                }
                at_gt += __popc(mg);
                at_eq += __popc(me);
            }
        } else {
            // exact fallback over every visible pool, re-reading the scores for each bit
            const int64_t n_iter = (nv + kSel2Threads - 1) / kSel2Threads * kSel2Threads;
            uint32_t prefix = 0;
            for (int bit = 31; bit >= 0; --bit) {
                const uint32_t cand = prefix | (1u << bit);
                uint32_t c = 0;
                for (int64_t i = threadIdx.x; i < nv; i += kSel2Threads) c += order_key(row[i]) >= cand;
                if (block_count(c, red, bit & 1) >= (uint32_t)kTopPools) prefix = cand;
            }
            // keys above prefix, then ties in pool order, placed by per-warp counts (deterministic)
            uint32_t wgt = 0, weq = 0;
            for (int64_t i = threadIdx.x; i < n_iter; i += kSel2Threads) {
                const uint32_t uu = i < nv ? order_key(row[i]) : 0u;
                wgt += __popc(__ballot_sync(0xffffffffu, i < nv && uu > prefix));
                weq += __popc(__ballot_sync(0xffffffffu, i < nv && uu == prefix));
            }
            uint32_t n_gt = 0, n_eq_unused = 0;
            uint32_t at_gt = warp_offsets(wgt, red, n_gt);
            uint32_t at_eq = n_gt + warp_offsets(weq, red + 16, n_eq_unused);
            const uint32_t below = (1u << lane) - 1;
            for (int64_t i = threadIdx.x; i < n_iter; i += kSel2Threads) {
                const uint32_t uu = i < nv ? order_key(row[i]) : 0u;
                const bool pg = i < nv && uu > prefix, pe = i < nv && uu == prefix;
                const uint32_t mg = __ballot_sync(0xffffffffu, pg), me = __ballot_sync(0xffffffffu, pe);
                const uint32_t sg = at_gt + __popc(mg & below), se = at_eq + __popc(me & below);
                if (pg) {
                    key[sg] = key_to_float(uu);
                    idx[sg] = (int32_t)i;
                }
                if (pe && se < (uint32_t)kTopPools) {
                    key[se] = key_to_float(uu);
                    idx[se] = (int32_t)i;
                }
                at_gt += __popc(mg);
                at_eq += __popc(me);
            }
        }
    }
}

// selected pools idx[0, ntop) -> top row, their cells plus the incomplete tail (pos, pos-1, ...) -> sel row
__device__ void write_selection(int t, int64_t pos, const int32_t * idx, int32_t * __restrict__ top,
                                int32_t * __restrict__ sel, int32_t * __restrict__ nsel) {
    const int64_t nv = (pos + 1) / kPool;
    const int ntop = (int)(nv < kTopPools ? nv : kTopPools);
    int32_t * trow = top + (int64_t)t * kTopPools;
    int32_t * srow = sel + (int64_t)t * kSelLd;
    for (int i = threadIdx.x; i < kTopPools * kPool; i += blockDim.x) {
        const int pi = i / kPool;
        if (pi < ntop) srow[i] = idx[pi] * kPool + (i % kPool);
        if ((i % kPool) == 0) trow[pi] = pi < ntop ? idx[pi] : -1;
    }
    const int n_tail = (int)((pos + 1) % kPool);
    if (threadIdx.x < n_tail) srow[ntop * kPool + threadIdx.x] = (int32_t)(pos - threadIdx.x);
    if (threadIdx.x == 0) nsel[t] = ntop * kPool + n_tail;
}

__global__ void __launch_bounds__(kSel2Threads)
select_cells_fast(const float * __restrict__ score, int64_t ld, const int64_t * __restrict__ dpos, int64_t pos_off,
                  int32_t * __restrict__ top, int32_t * __restrict__ sel, int32_t * __restrict__ nsel) {
    __shared__ float key[kTopPools];
    __shared__ int32_t idx[kTopPools];
    const int t = blockIdx.x;
    const int64_t pos = batch_pos0(dpos, pos_off) + t;
    top_pools(score + (int64_t)t * ld, (pos + 1) / kPool, true, key, idx);
    __syncthreads();
    write_selection(t, pos, idx, top, sel, nsel);
}

// Two-stage selection for decode / verify at long context (one block per token leaves the GPU idle): block
// (slice, token) keeps the top 512 of its kSelSlice pools (all of them if fewer), densely at slice * 512 of the
// token's candidate row; the last block of the token to finish (per-token counter in the cache, reset by that
// block) takes the top 512 of the candidates.
constexpr int kSelSlice = kSelSlicePools;

__global__ void __launch_bounds__(kSel2Threads)
select_two_stage(const float * __restrict__ score, int64_t ld, const int64_t * __restrict__ dpos, int64_t pos_off,
                 float * __restrict__ ckey, int32_t * __restrict__ cidx, int64_t cld, unsigned int * __restrict__ done,
                 int32_t * __restrict__ top, int32_t * __restrict__ sel, int32_t * __restrict__ nsel) {
    __shared__ float key[kTopPools];
    __shared__ int32_t idx[kTopPools];
    __shared__ bool s_last;
    const int t = blockIdx.y;
    const int64_t pos = batch_pos0(dpos, pos_off) + t;
    const int64_t nv = (pos + 1) / kPool;
    const int64_t s0 = (int64_t)blockIdx.x * kSelSlice;
    if (s0 < nv) {
        const int64_t nk = min((int64_t)kSelSlice, nv - s0);
        top_pools(score + (int64_t)t * ld + s0, nk, false, key, idx);
        __syncthreads();
        const int n = (int)min(nk, (int64_t)kTopPools);
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            ckey[t * cld + blockIdx.x * kTopPools + i] = key[i];
            cidx[t * cld + blockIdx.x * kTopPools + i] = (int32_t)(s0 + idx[i]);
        }
        __threadfence();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        s_last = atomicAdd(&done[t], 1u) == gridDim.x - 1;
        if (s_last) done[t] = 0;   // ready for the next launch
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const int64_t nu = (nv + kSelSlice - 1) / kSelSlice;   // slices with pools
    const int64_t nk = nu ? (nu - 1) * kTopPools + min((int64_t)kTopPools, nv - (nu - 1) * kSelSlice) : 0;
    // nv <= 512: slice 0 holds every pool in order, so the sorted result matches select_cells_fast
    top_pools(ckey + t * cld, nk, true, key, idx);
    __syncthreads();
    const int ntop = (int)(nv < kTopPools ? nv : kTopPools);
    for (int i = threadIdx.x; i < ntop; i += blockDim.x) idx[i] = __ldcg(cidx + t * cld + idx[i]);
    __syncthreads();
    write_selection(t, pos, idx, top, sel, nsel);
}

// ---- sparse latent attention ------------------------------------------------------------------------------
constexpr int kKLd = kR + 8;   // padded F16 row of the staged latent

// Staging of ROWS latent rows (cells[r], r < cnt; zero rows past cnt) into shared memory with row stride kKLd, as
// a register load (all index loads, then all row loads, before any use: one round trip each instead of a chain)
// and a shared-memory store, so the next tile can be loaded during compute.
template <int ROWS>
struct RowRegs {
    static constexpr int kChunks = kR / 8;              // uint4 per row
    static constexpr int kPer = ROWS * kChunks / 256;   // per thread, blockDim == 256
    uint4 v[kPer];
};

template <int ROWS>
__device__ __forceinline__ void load_rows(RowRegs<ROWS> & rr, const LatentRows & latent,
                                          const int32_t * __restrict__ cells, int cnt) {
    constexpr int kChunks = RowRegs<ROWS>::kChunks, kPer = RowRegs<ROWS>::kPer;
    int32_t cell[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int r = (threadIdx.x + j * 256) / kChunks;
        cell[j] = r < cnt ? cells[r] : -1;
    }
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int ch = (threadIdx.x + j * 256) % kChunks;
        rr.v[j] = cell[j] >= 0 ? latent_load8(latent, cell[j], ch * 8) : make_uint4(0, 0, 0, 0);
    }
}

template <int ROWS>
__device__ __forceinline__ void store_rows(__half * ks, const RowRegs<ROWS> & rr) {
    constexpr int kChunks = RowRegs<ROWS>::kChunks, kPer = RowRegs<ROWS>::kPer;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int i = threadIdx.x + j * 256;
        *reinterpret_cast<uint4 *>(ks + (i / kChunks) * kKLd + (i % kChunks) * 8) = rr.v[j];
    }
}

// o[t][h] = sum_s exp(m_s - M) O_s / sum_s exp(m_s - M) l_s. Block = (head, token): warp 0 turns the (m, l) of
// every split into weights, then kCbGroups groups of kR/4 threads each sum every kCbGroups-th split (more loads in
// flight than one thread per float4 walking all splits), merged through shared memory.
constexpr int kCbGroups = 4;
constexpr int kCbMaxSplit = 64;
constexpr int kCbThreads = kCbGroups * kR / 4;
static_assert(kMaxSplits <= kCbMaxSplit, "split weights must fit one warp's two passes");

__global__ void __launch_bounds__(kCbThreads)
attn_combine(const float * __restrict__ part_o, const float2 * __restrict__ part_ml, const int32_t * __restrict__ nsel,
             int split, int n_split, float * __restrict__ o, void * __restrict__ prepared) {
    __shared__ float wsh[kCbMaxSplit];
    __shared__ float inv_l;
    __shared__ float4 red[kCbGroups - 1][kR / 4];
    const int h = blockIdx.x, t = blockIdx.y;
    const int ns = (nsel[t] + split - 1) / split;
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        float2 ml[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int s2 = lane + 32 * j;
            ml[j] = s2 < ns ? part_ml[((int64_t)t * n_split + s2) * kH + h] : make_float2(-INFINITY, 0.f);
        }
        const float M = warp_max(fmaxf(ml[0].x, ml[1].x));
        float L = 0.f;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int s2 = lane + 32 * j;
            if (s2 < ns) {
                const float e = __expf(ml[j].x - M);
                wsh[s2] = e;
                L += e * ml[j].y;
            }
        }
        L = warp_sum(L);
        if (lane == 0) inv_l = 1.f / L;
    }
    __syncthreads();
    const int g = threadIdx.x / (kR / 4), d4 = threadIdx.x % (kR / 4);
    const float4 * po = reinterpret_cast<const float4 *>(part_o) + ((int64_t)t * n_split * kH + h) * (kR / 4) + d4;
    const int64_t stride = (int64_t)kH * (kR / 4);   // float4 between consecutive splits of one head
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
    for (int s2 = g; s2 < ns; s2 += kCbGroups) {
        const float e = wsh[s2];
        const float4 v = po[s2 * stride];
        acc.x += e * v.x;
        acc.y += e * v.y;
        acc.z += e * v.z;
        acc.w += e * v.w;
    }
    if (g) red[g - 1][d4] = acc;
    __syncthreads();
    if (g) return;
#pragma unroll
    for (int k = 0; k < kCbGroups - 1; ++k) {
        const float4 v = red[k][d4];
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }
    const float inv = inv_l;
    const float4 r = make_float4(acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv);
    reinterpret_cast<float4 *>(o + ((int64_t)t * kH + h) * kR)[d4] = r;
    if (!prepared) return;
    // Wv_b's Q8 prepared input: per-head batch, column h * T + t; 8 threads hold one 32-element block
    const int64_t T = gridDim.y, col = (int64_t)h * T + t;
    const float amax = q8_group_amax<8>(fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)), fmaxf(fabsf(r.z), fabsf(r.w))));
    const float id = q8_inv_scale(amax);
    char4 qv;
    qv.x = q8_quant(r.x, id);
    qv.y = q8_quant(r.y, id);
    qv.z = q8_quant(r.z, id);
    qv.w = q8_quant(r.w, id);
    reinterpret_cast<char4 *>(static_cast<int8_t *>(prepared) + col * kR)[d4] = qv;
    if ((d4 & 7) == 0)
        reinterpret_cast<float *>(static_cast<char *>(prepared) + q8_scales_offset(kH * T, kR))[col * (kR / 32) + d4 / 8] =
            q8_scale(amax);
}

// ---- sparse latent attention, prefill (online softmax over all selected cells) ------------------------------
// Block = (32 heads, token), 8 warps; every staged latent row serves 32 heads. Q sits in shared memory; cells are
// staged 32 at a time (the next tile is prefetched into registers during compute). S: warp -> head tile (w & 1) and
// k quarter (w >> 1) of 128 dims, whose 8 Q fragments stay in registers; both cell tiles, the four quarters' partial
// S in separate buffers. Online softmax per head;
// the running O stays in registers (warp -> both head tiles x 4 dim tiles at w * 64) and is rescaled in
// place: the row of each accumulator element is read once from a fragment loaded off a row-index matrix.
constexpr int kPfTile = 32;
constexpr int kPfHeads = 32;
constexpr int kSpLd = kPfTile + 4;
constexpr int kPpLd = kPfTile + 8;
constexpr int kPfOsLd = kR + 4;
constexpr size_t kPfQK = sizeof(__half) * (kPfHeads + kPfTile) * kKLd;   // Q rows, then the staged cells
constexpr size_t kPfSmem = kPfQK + sizeof(float) * 4 * kPfHeads * kSpLd + sizeof(__half) * kPfHeads * kPpLd +
                           sizeof(float) * 3 * kPfHeads;
static_assert(sizeof(float) * kPfHeads * kPfOsLd <= kPfQK, "O staging must fit over Q and the tile");

// Block (sp, hg, t) covers cells [sp * split, (sp + 1) * split) of token t. Without part_o (one split over all cells)
// it writes o normalized; with part_o it writes the split's unnormalized O and (m, l) for attn_combine.
__global__ void __launch_bounds__(256)
attn_flash(const __half * __restrict__ qh, const LatentRows latent, const int32_t * __restrict__ sel,
           const int32_t * __restrict__ nsel, int sel_ld, int split, int n_split, float * __restrict__ o,
           float * __restrict__ part_o, float2 * __restrict__ part_ml) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * qs = reinterpret_cast<__half *>(smem_raw);
    __half * ks = qs + kPfHeads * kKLd;
    float * sp = reinterpret_cast<float *>(smem_raw + kPfQK);   // [4][32][kSpLd]
    __half * pp = reinterpret_cast<__half *>(sp + 4 * kPfHeads * kSpLd);
    float * m_run = reinterpret_cast<float *>(pp + kPfHeads * kPpLd);
    float * l_run = m_run + kPfHeads;
    float * alpha = l_run + kPfHeads;
    const int t = blockIdx.z, hg = blockIdx.y, sp_i = blockIdx.x;
    const int c_beg = sp_i * split;
    const int n_all = nsel[t];
    if (c_beg >= n_all) return;
    const int n = min(split, n_all - c_beg);   // cells of this block
    const int32_t * cells = sel + (int64_t)t * sel_ld + c_beg;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

    // Q rows and the accumulator row map (sp[r][c] = r, loaded as an accumulator fragment)
    const __half * q = qh + ((int64_t)t * kH + hg * kPfHeads) * kR;
    for (int i = threadIdx.x; i < kPfHeads * kR / 8; i += blockDim.x) {
        const int r = i / (kR / 8), ch = i % (kR / 8);
        *reinterpret_cast<uint4 *>(qs + r * kKLd + ch * 8) = *reinterpret_cast<const uint4 *>(q + r * kR + ch * 8);
    }
    sp[threadIdx.x] = (float)(threadIdx.x / 16);
    if (threadIdx.x < kPfHeads) {
        m_run[threadIdx.x] = -INFINITY;
        l_run[threadIdx.x] = 0.f;
    }
    __syncthreads();
    using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    Acc rowf;
    wmma::load_matrix_sync(rowf, sp, 16, wmma::mem_row_major);
    const int ht = warp & 1, kq = warp >> 1;   // S: head tile, k quarter; P V: dims [warp * 64, + 64), both head tiles
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qa[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) wmma::load_matrix_sync(qa[k], qs + ht * 16 * kKLd + kq * 128 + k * 16, kKLd);
    Acc ov[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) wmma::fill_fragment(ov[j], 0.f);

    RowRegs<kPfTile> nxt;
    load_rows<kPfTile>(nxt, latent, cells, min(kPfTile, n));
    for (int c0 = 0; c0 < n; c0 += kPfTile) {
        const int cnt = min(kPfTile, n - c0);
        __syncthreads();   // previous tile's P V (and the row map load) done with ks / sp
        store_rows<kPfTile>(ks, nxt);
        __syncthreads();
        if (c0 + kPfTile < n) load_rows<kPfTile>(nxt, latent, cells + c0 + kPfTile, min(kPfTile, n - c0 - kPfTile));
        {
            Acc acc[2];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) wmma::fill_fragment(acc[nt], 0.f);
#pragma unroll
            for (int k = 0; k < 8; ++k)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                    wmma::load_matrix_sync(b, ks + nt * 16 * kKLd + kq * 128 + k * 16, kKLd);
                    wmma::mma_sync(acc[nt], qa[k], b, acc[nt]);
                }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                wmma::store_matrix_sync(sp + (kq * kPfHeads + ht * 16) * kSpLd + nt * 16, acc[nt], kSpLd,
                                        wmma::mem_row_major);
        }
        __syncthreads();
        // online softmax: warp -> heads 4w .. 4w+3, 8 lanes per head, 4 cells per lane
        {
            const int h = warp * 4 + (lane >> 3), c = (lane & 7) * 4;
            float sv[4];
            {
                float4 a = *reinterpret_cast<const float4 *>(sp + h * kSpLd + c);
#pragma unroll
                for (int g2 = 1; g2 < 4; ++g2) {
                    const float4 b = *reinterpret_cast<const float4 *>(sp + (g2 * kPfHeads + h) * kSpLd + c);
                    a.x += b.x;
                    a.y += b.y;
                    a.z += b.z;
                    a.w += b.w;
                }
                sv[0] = a.x;
                sv[1] = a.y;
                sv[2] = a.z;
                sv[3] = a.w;
            }
            float mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (c + i >= cnt) sv[i] = -INFINITY;
                mx = fmaxf(mx, sv[i]);
            }
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float m_old = m_run[h];
            const float m_new = fmaxf(m_old, mx);
            __half ph[4];
            float lsum = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ph[i] = __float2half(c + i < cnt ? __expf(sv[i] - m_new) : 0.f);
                lsum += __half2float(ph[i]);
            }
            *reinterpret_cast<uint2 *>(pp + h * kPpLd + c) = *reinterpret_cast<const uint2 *>(ph);
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
            __syncwarp();
            if ((lane & 7) == 0) {
                const float a = __expf(m_old - m_new);
                alpha[h] = a;
                l_run[h] = l_run[h] * a + lsum;
                m_run[h] = m_new;
            }
        }
        __syncthreads();
        // O = alpha O + P V (ov[j]: head tile j >> 2, dim tile warp * 4 + (j & 3); each V fragment serves both
        // head tiles, which halves the shared-memory fragment traffic)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int e = 0; e < ov[j].num_elements; ++e) ov[j].x[e] *= alpha[(j >> 2) * 16 + (int)rowf.x[e]];
#pragma unroll
        for (int k = 0; k < kPfTile / 16; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0, a1;
            wmma::load_matrix_sync(a0, pp + k * 16, kPpLd);
            wmma::load_matrix_sync(a1, pp + 16 * kPpLd + k * 16, kPpLd);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                wmma::load_matrix_sync(b, ks + k * 16 * kKLd + (warp * 4 + j) * 16, kKLd);
                wmma::mma_sync(ov[j], a0, b, ov[j]);
                wmma::mma_sync(ov[4 + j], a1, b, ov[4 + j]);
            }
        }
    }
    // O (/ l for the final output) through shared memory (Q and the staged rows are dead), then coalesced rows
    __syncthreads();
    float * os = reinterpret_cast<float *>(smem_raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (!part_o)
#pragma unroll
            for (int e = 0; e < ov[j].num_elements; ++e) ov[j].x[e] /= l_run[(j >> 2) * 16 + (int)rowf.x[e]];
        wmma::store_matrix_sync(os + (j >> 2) * 16 * kPfOsLd + (warp * 4 + (j & 3)) * 16, ov[j], kPfOsLd,
                                wmma::mem_row_major);
    }
    if (part_o && threadIdx.x < kPfHeads)
        part_ml[((int64_t)t * n_split + sp_i) * kH + hg * kPfHeads + threadIdx.x] =
            make_float2(m_run[threadIdx.x], l_run[threadIdx.x]);
    __syncthreads();
    float4 * dst = reinterpret_cast<float4 *>(
        part_o ? part_o + (((int64_t)t * n_split + sp_i) * kH + hg * kPfHeads) * kR
               : o + ((int64_t)t * kH + hg * kPfHeads) * kR);
    for (int i = threadIdx.x; i < kPfHeads * kR / 4; i += blockDim.x) {
        const int h = i / (kR / 4), d4 = i % (kR / 4);
        dst[i] = *reinterpret_cast<const float4 *>(os + h * kPfOsLd + d4 * 4);
    }
}

__global__ void fill_random_f16(__half * p, int64_t n, uint64_t seed) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // splitmix64 -> sum of 4 uniforms, centred and scaled to unit variance (Irwin-Hall)
    uint64_t z = seed + (uint64_t)i * 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    float u = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) u += (float)((z >> (16 * k)) & 0xffff) / 65535.f;
    p[i] = __float2half((u - 2.f) * 1.7320508f);
}

// Q8 latent rows of the same N(0,1)-like values: int8 at a fixed scale of 1/32 (|x| < 3.97)
__global__ void fill_random_q8(LatentRows L, int64_t n, uint64_t seed) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t z = seed + (uint64_t)i * 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    float u = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) u += (float)((z >> (16 * k)) & 0xffff) / 65535.f;
    L.q8[i] = (int8_t)fminf(fmaxf(rintf((u - 2.f) * 1.7320508f * 32.f), -127.f), 127.f);
    if (i % kQ8Group == 0) L.scale[i / kQ8Group] = __float2half(1.f / 32.f);
}

// rows [pos0, pos0 + n) as F16 (one block of kR / 8 threads per row)
__global__ void latent_to_f16(LatentRows L, int64_t pos0, __half * __restrict__ out) {
    const int64_t r = blockIdx.x;
    *reinterpret_cast<uint4 *>(out + r * kR + threadIdx.x * 8) = latent_load8(L, pos0 + r, threadIdx.x * 8);
}

// per device, before any capture (mla_cache_create calls it)
void set_kernel_attributes() {
    static bool done[64] = {};
    int dev = 0;
    MLA_CUDA(cudaGetDevice(&dev));
    if (dev < 64 && done[dev]) return;
    MLA_CUDA(cudaFuncSetAttribute(attn_flash, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kPfSmem));
    if (dev < 64) done[dev] = true;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------

struct MlaCache {
    int64_t cap = 0;
    int64_t n = 0;
    MlaLatent latent_type = MlaLatent::F16;
    LatentRows latent{};         // F16 [cap][512], or int8 [cap][512] + F16 scales [cap][8]
    KgRows kg{};                 // ring [kKgRing][256] + tags on the device, host copy [cap][256]
    __half * kg_host = nullptr;  // host pointer of kg.host
    int64_t kg_hole_a = 0, kg_hole_b = 0;   // host key | gate rows [a, b) not held (imported ranges), empty if a == b
    __half * pooled = nullptr;   // [cap/4][128]
    unsigned int * sel_done = nullptr;   // [kSplitMaxT] two-stage selection: slices finished per token (0 between calls)
    // side stream (highest priority) for the indexer chain or Wk_b, concurrent with the main stream's projections
    cudaStream_t side = nullptr;
    cudaEvent_t ev_fork = nullptr, ev_join = nullptr, ev_iw = nullptr;
    // last sub-chunk processed, for mla_taps
    int last_T = 0, last_TC = 0;
    int64_t last_pos0 = 0;
};

namespace {
size_t latent_row_bytes(MlaLatent t) {
    return t == MlaLatent::Q8 ? kR + sizeof(__half) * kR / kQ8Group : sizeof(__half) * kR;
}

// whether the host key | gate rows of the incomplete pool at m ([4 (m/4), m)) are held
bool kg_tail_held(const MlaCache * c, int64_t m) {
    const int64_t lo = m / kPool * kPool;
    return lo == m || m <= c->kg_hole_a || lo >= c->kg_hole_b;
}

// the sequence is cut to m (or appended to at m): positions >= m are rewritten before any pool reads them
void kg_cut(MlaCache * c, int64_t m) {
    if (!kg_tail_held(c, m)) {
        std::fprintf(stderr, "mla: cut at %lld needs key | gate rows of an imported range [%lld, %lld)\n",
                     (long long)m, (long long)c->kg_hole_a, (long long)c->kg_hole_b);
        std::abort();
    }
    c->kg_hole_b = std::min(c->kg_hole_b, m);
    if (c->kg_hole_b <= c->kg_hole_a) c->kg_hole_a = c->kg_hole_b = 0;
}
}  // namespace

size_t mla_cache_bytes(const Geometry & g, int64_t max_ctx, MlaLatent latent) {
    (void)g;
    return latent_row_bytes(latent) * (size_t)max_ctx +
           sizeof(__half) * (size_t)(kKgRing * 2 * kID + (max_ctx / kPool + kPooledPad) * kID) +
           sizeof(int64_t) * kKgRing;
}

size_t mla_cache_host_bytes(int64_t max_ctx) { return sizeof(__half) * (size_t)max_ctx * 2 * kID; }

MlaCache * mla_cache_create(const Geometry & g, int64_t max_ctx, MlaLatent latent) {
    check_geometry(g);
    auto * c = new MlaCache;
    c->cap = max_ctx;
    c->latent_type = latent;
    if (latent == MlaLatent::Q8) {
        MLA_CUDA(cudaMalloc(&c->latent.q8, (size_t)max_ctx * kR));
        MLA_CUDA(cudaMalloc(&c->latent.scale, sizeof(__half) * max_ctx * (kR / kQ8Group)));
    } else {
        MLA_CUDA(cudaMalloc(&c->latent.f16, sizeof(__half) * max_ctx * kR));
    }
    MLA_CUDA(cudaMalloc(&c->kg.ring, sizeof(__half) * kKgRing * 2 * kID));
    MLA_CUDA(cudaMalloc(&c->kg.tag, sizeof(int64_t) * kKgRing));
    MLA_CUDA(cudaMemset(c->kg.tag, 0xff, sizeof(int64_t) * kKgRing));
    MLA_CUDA(cudaHostAlloc(&c->kg_host, mla_cache_host_bytes(max_ctx), cudaHostAllocMapped));
    MLA_CUDA(cudaHostGetDevicePointer(&c->kg.host, c->kg_host, 0));
    MLA_CUDA(cudaMalloc(&c->pooled, sizeof(__half) * (max_ctx / kPool + kPooledPad) * kID));
    MLA_CUDA(cudaMemset(c->pooled, 0, sizeof(__half) * (max_ctx / kPool + kPooledPad) * kID));
    MLA_CUDA(cudaMalloc(&c->sel_done, sizeof(unsigned int) * kSplitMaxT));
    MLA_CUDA(cudaMemset(c->sel_done, 0, sizeof(unsigned int) * kSplitMaxT));
    int prio_least = 0, prio_greatest = 0;
    MLA_CUDA(cudaDeviceGetStreamPriorityRange(&prio_least, &prio_greatest));
    MLA_CUDA(cudaStreamCreateWithPriority(&c->side, cudaStreamNonBlocking, prio_greatest));
    MLA_CUDA(cudaEventCreateWithFlags(&c->ev_fork, cudaEventDisableTiming));
    MLA_CUDA(cudaEventCreateWithFlags(&c->ev_join, cudaEventDisableTiming));
    MLA_CUDA(cudaEventCreateWithFlags(&c->ev_iw, cudaEventDisableTiming));
    set_kernel_attributes();
    return c;
}

void mla_cache_destroy(MlaCache * c) {
    if (!c) return;
    cudaFree(c->latent.f16);
    cudaFree(c->latent.q8);
    cudaFree(c->latent.scale);
    cudaFree(c->kg.ring);
    cudaFree(c->kg.tag);
    cudaFreeHost(c->kg_host);
    cudaFree(c->pooled);
    cudaFree(c->sel_done);
    cudaEventDestroy(c->ev_fork);
    cudaEventDestroy(c->ev_join);
    cudaEventDestroy(c->ev_iw);
    cudaStreamDestroy(c->side);
    delete c;
}

void mla_cache_truncate(MlaCache * c, int64_t n_tokens) {
    if (n_tokens < 0 || n_tokens > c->n) {
        std::fprintf(stderr, "mla: truncate to %lld outside [0, %lld]\n", (long long)n_tokens, (long long)c->n);
        std::abort();
    }
    kg_cut(c, n_tokens);
    c->n = n_tokens;   // rows and pools at or past the cut are dead; an append that completes a pool re-pools it
}

bool mla_cache_can_truncate(const MlaCache * c, int64_t n_tokens) {
    return n_tokens >= 0 && n_tokens <= c->n && kg_tail_held(c, n_tokens);
}

int64_t mla_cache_size(const MlaCache * c) { return c->n; }
int64_t mla_cache_capacity(const MlaCache * c) { return c->cap; }
const void * mla_cache_pooled(const MlaCache * c) { return c->pooled; }
MlaLatent mla_cache_latent_type(const MlaCache * c) { return c->latent_type; }
const void * mla_cache_latent(const MlaCache * c) {
    return c->latent.q8 ? static_cast<const void *>(c->latent.q8) : c->latent.f16;
}
const void * mla_cache_latent_scale(const MlaCache * c) { return c->latent.scale; }

void mla_cache_latent_f16(const MlaCache * c, int64_t pos0, int64_t n, void * out, cudaStream_t s) {
    if (pos0 < 0 || n < 0 || pos0 + n > c->cap) {
        std::fprintf(stderr, "mla: latent rows [%lld, +%lld) outside the capacity %lld\n", (long long)pos0,
                     (long long)n, (long long)c->cap);
        std::abort();
    }
    if (n > 0) latent_to_f16<<<(unsigned)n, kR / 8, 0, s>>>(c->latent, pos0, static_cast<__half *>(out));
    MLA_CUDA(cudaGetLastError());
}
const void * mla_cache_kg(const MlaCache * c) { return c->kg.host; }

void mla_cache_fill_random(MlaCache * c, const MlaWeights & w, int64_t n, uint64_t seed, cudaStream_t s) {
    if (n > c->cap) n = c->cap;
    const int64_t nl = n * kR, nk = n * 2 * kID;
    if (c->latent.q8)
        fill_random_q8<<<(unsigned)((nl + 255) / 256), 256, 0, s>>>(c->latent, nl, seed);
    else
        fill_random_f16<<<(unsigned)((nl + 255) / 256), 256, 0, s>>>(c->latent.f16, nl, seed);
    // straight into the host copy; the ring forgets every row (pool_new and later pools read the host copy)
    fill_random_f16<<<(unsigned)((nk + 255) / 256), 256, 0, s>>>(c->kg.host, nk, seed ^ 0x5555555555555555ull);
    MLA_CUDA(cudaMemsetAsync(c->kg.tag, 0xff, sizeof(int64_t) * kKgRing, s));
    if (n / kPool > 0)
        pool_new<<<(unsigned)(n / kPool), kID, 0, s>>>(c->kg, (const float *)w.idx_ape.data, nullptr, 0, n, c->cap,
                                                       c->pooled);
    MLA_CUDA(cudaGetLastError());
    c->n = n;
    c->kg_hole_a = c->kg_hole_b = 0;
}

size_t mla_scratch_bytes(const Geometry & g, int max_T, int64_t max_ctx) {
    return layout(std::min(std::max(max_T, 1), kChunk), max_ctx, g.n_embd).total;
}

MlaTaps mla_taps(const Geometry & g, const MlaCache * c, void * scratch) {
    const Layout L = layout(c->last_TC, c->cap, g.n_embd);
    auto * base = static_cast<char *>(scratch);
    MlaTaps tp;
    tp.T = c->last_T;
    tp.pos0 = c->last_pos0;
    tp.n_pool_ld = (int)L.n_pool_ld;
    tp.qr = reinterpret_cast<const float *>(base + L.qr);
    tp.q_abs = reinterpret_cast<const float *>(base + L.qabs);
    tp.kv = reinterpret_cast<const float *>(base + L.kv);
    tp.iq = reinterpret_cast<const float *>(base + L.iq);
    tp.ik = reinterpret_cast<const float *>(base + L.ik);
    tp.ig = reinterpret_cast<const float *>(base + L.ig);
    tp.iw = reinterpret_cast<const float *>(base + L.iw);
    tp.score = reinterpret_cast<const float *>(base + L.score);
    tp.sel = reinterpret_cast<const int32_t *>(base + L.sel);
    tp.nsel = reinterpret_cast<const int32_t *>(base + L.nsel);
    tp.sel_ld = kSelLd;
    tp.top = reinterpret_cast<const int32_t *>(base + L.top);
    tp.o = reinterpret_cast<const float *>(base + L.o);
    tp.v = reinterpret_cast<const float *>(base + L.q);
    return tp;
}

namespace {


// One sub-chunk of T <= TC tokens at position pos_off (+ *dpos when dpos is set). With dpos every launch
// configuration depends only on T and the cache capacity (capturable); blocks past the live length exit early.
// With shared (sel, nsel, sel_ld), the indexer scoring and selection are skipped and attention runs over the given
// cells (rows already offset to this chunk).
void forward_chunk(const Geometry & g, const MlaWeights & w, MlaCache * c, const Layout & L, int T,
                   const int64_t * dpos, int64_t pos_off, const float * x, float * out, char * base, cudaStream_t s,
                   const MlaSelection * shared, const MatmulInput * xin) {
    float * qr = reinterpret_cast<float *>(base + L.qr);
    float * q = reinterpret_cast<float *>(base + L.q);
    float * qabs = reinterpret_cast<float *>(base + L.qabs);
    float * o = reinterpret_cast<float *>(base + L.o);
    float * kv = reinterpret_cast<float *>(base + L.kv);
    float * iq = reinterpret_cast<float *>(base + L.iq);
    float * ik = reinterpret_cast<float *>(base + L.ik);
    float * ig = reinterpret_cast<float *>(base + L.ig);
    float * iw = reinterpret_cast<float *>(base + L.iw);
    float * score = reinterpret_cast<float *>(base + L.score);
    int32_t * top = reinterpret_cast<int32_t *>(base + L.top);
    int32_t * sel_own = reinterpret_cast<int32_t *>(base + L.sel);
    int32_t * nsel_own = reinterpret_cast<int32_t *>(base + L.nsel);
    const int32_t * sel = shared ? shared->sel : sel_own;
    const int32_t * nsel = shared ? shared->nsel : nsel_own;
    const int sel_ld = shared ? shared->sel_ld : kSelLd;
    __half * qh = reinterpret_cast<__half *>(base + L.qh);
    __half * iqh = reinterpret_cast<__half *>(base + L.iqh);
    float * part_o = reinterpret_cast<float *>(base + L.part_o);
    float2 * part_ml = reinterpret_cast<float2 *>(base + L.part_ml);
    static const bool naive = std::getenv("MLA_NAIVE") && std::atoi(std::getenv("MLA_NAIVE")) != 0;
    const int D = g.n_embd;
    void * mm_in = base + L.mm_in;
    void * mm_w = base + L.mm_w;
    auto check_ws = [&](const WeightRef & W, int TT) {
        if (matmul_run_bytes(W, TT) > L.mm_w_bytes) {
            std::fprintf(stderr, "mla: weight type %d needs more matmul scratch than reserved\n", W.type);
            std::abort();
        }
    };
    auto run_on = [&](cudaStream_t st, const WeightRef & W, const MatmulInput & in, float * y, int64_t ldy,
                      int64_t y_batch_stride) {
        check_ws(W, in.T);
        matmul_run(W, in, y, mm_w, L.mm_w_bytes, st, ldy, y_batch_stride);
    };
    auto run = [&](const WeightRef & W, const MatmulInput & in, float * y, int64_t ldy, int64_t y_batch_stride) {
        run_on(s, W, in, y, ldy, y_batch_stride);
    };
    // batch-1 weights (at most 8) on one input: one matmul_run_multi launch per weight type present, up to 4
    // weights each (the final pack mixes Q8_0R, BF16 and F32 projections)
    auto run_group = [&](std::initializer_list<const WeightRef *> ws, std::initializer_list<float *> ys,
                         std::initializer_list<int64_t> ldys, const MatmulInput & in) {
        const WeightRef * wa[8];
        float * ya[8];
        int64_t la[8];
        bool done[8] = {};
        const int n = (int)ws.size();
        std::copy(ws.begin(), ws.end(), wa);
        std::copy(ys.begin(), ys.end(), ya);
        std::copy(ldys.begin(), ldys.end(), la);
        for (int i = 0; i < n; ++i) {
            if (done[i]) continue;
            WeightRef wv[4];
            float * yv[4];
            int64_t lv[4];
            int m = 0;
            for (int j = i; j < n; ++j) {
                if (done[j] || wa[j]->type != wa[i]->type || m == 4) continue;
                check_ws(*wa[j], in.T);
                wv[m] = *wa[j];
                yv[m] = ya[j];
                lv[m++] = la[j];
                done[j] = true;
            }
            matmul_run_multi(wv, yv, lv, m, in, mm_w, L.mm_w_bytes, s);
        }
    };
    const bool fast = !(naive && !dpos);
    // GEMV path: the indexer head projection (a small, latency-bound F32 GEMV) runs on the side stream, and the
    // q_resid RMSNorm and attn_combine write the next projection's Q8 input themselves (into mm_in2)
    const bool fused = fast && L.mm_in2_bytes > 0 && matmul_input_is_q8(kQr, T) && matmul_input_is_q8(kR, T);
    void * mm_in2 = base + L.mm_in2;
    static_assert(kQr % 256 == 0, "rmsnorm_rows_q8 needs whole 256-thread steps");

    // projections of x (shared prepared input), then of q_resid
    const MatmulInput in_x = xin ? *xin : matmul_prepare(x, D, T, 1, D, 0, mm_in, s);
    const float iw_scale = 1.f / std::sqrt((float)(kIH * kID));
    auto fork = [&]() {
        MLA_CUDA(cudaEventRecord(c->ev_fork, s));
        MLA_CUDA(cudaStreamWaitEvent(c->side, c->ev_fork, 0));
    };
    auto norm_kv = [&](cudaStream_t st) {
        rmsnorm_rows<<<T, 256, 0, st>>>(kv, kR, (const float *)w.kv_a_norm.data, g.rms_eps);
        layernorm_rows<<<T, kID, 0, st>>>(ik, kID, (const float *)w.idx_k_norm_w.data,
                                          (const float *)w.idx_k_norm_b.data, g.ln_eps);
    };
    auto write_pools = [&](cudaStream_t st) {
        // pools completed by this chunk: at most (T + 3) / 4, exactly that many blocks when the position is on device
        const int64_t n_pool_blocks = dpos ? (T + kPool - 1) / kPool : (pos_off + T) / kPool - pos_off / kPool;
        if (n_pool_blocks > 0)
            pool_new<<<(unsigned)n_pool_blocks, kID, 0, st>>>(c->kg, (const float *)w.idx_ape.data, dpos, pos_off, T,
                                                              c->cap, c->pooled);
    };
    auto write_caches = [&](cudaStream_t st) {
        cache_write<<<T, 256, 0, st>>>(kv, ik, ig, dpos, pos_off, c->cap, c->latent, c->kg);
        write_pools(st);
    };
    // pools visible to the last token (an upper bound from the capacity when the position is on device)
    const int64_t nv_max = dpos ? c->cap / kPool : (pos_off + T) / kPool;
    // indexer scoring and top-k selection (fast path; the shared selection skips them)
    auto score_select = [&](cudaStream_t st) {
        if (shared) return;
        const int64_t ni = (int64_t)T * kIH * kID;
        to_half_scaled<<<(unsigned)((ni / 4 + 255) / 256), 256, 0, st>>>(iq, ni, 1.f, iqh);
        if (nv_max > 0) {
            dim3 grid((unsigned)((nv_max + kScTile - 1) / kScTile), (T + kScTokens - 1) / kScTokens);
            score_wmma<<<grid, 256, 0, st>>>(iqh, iw, c->pooled, dpos, pos_off, T, L.n_pool_ld, score);
        }
        if (L.cand_ld) {
            float * ckey = reinterpret_cast<float *>(base + L.ckey);
            int32_t * cidx = reinterpret_cast<int32_t *>(base + L.cidx);
            const unsigned n_slices = (unsigned)(L.cand_ld / kTopPools);
            select_two_stage<<<dim3(n_slices, T), kSel2Threads, 0, st>>>(
                score, L.n_pool_ld, dpos, pos_off, ckey, cidx, L.cand_ld, c->sel_done, top, sel_own, nsel_own);
        } else {
            select_cells_fast<<<T, kSel2Threads, 0, st>>>(score, L.n_pool_ld, dpos, pos_off, top, sel_own, nsel_own);
        }
    };
    // q_abs[h] = Wk_b[h]^T q[h]: 64 per-head [256 -> 512] matrices over the [T][64][256] q (prepared into mm_in)
    const int64_t nq = (int64_t)T * kH * kR;
    auto q_abs = [&](cudaStream_t st) {
        run_on(st, w.k_b, matmul_prepare(q, kDq, T, kH, kH * kDq, kDq, mm_in, st), qabs, kH * kR, kR);
        if (fast)
            to_half_scaled<<<(unsigned)((nq / 4 + 255) / 256), 256, 0, st>>>(qabs, nq, 1.f / std::sqrt((float)kDq), qh);
    };

    if (fused) {
        // GEMV path. The cache's side stream (highest priority, so its small kernels are not queued behind the
        // GEMV blocks of the main stream) carries the latency-bound indexer chain: head projection, cache writes,
        // query projection, scoring and selection. The main stream carries the q path: q_a group, q_resid RMSNorm
        // (writing q_b's Q8 input into mm_in2), q_b, Wk_b. GEMVs use none of mm_w and nothing goes through cuBLAS
        // (whose workspace is shared); in_qr in mm_in2 is read-only until attn_combine, after the join.
        fork();
        run_on(c->side, w.idx_proj, in_x, iw, kIH, 0);
        scale_rows<<<(T * kIH + 255) / 256, 256, 0, c->side>>>(iw, (int64_t)T * kIH, iw_scale);
        MLA_CUDA(cudaEventRecord(c->ev_iw, c->side));
        run_group({&w.q_a, &w.kv_a, &w.idx_k, &w.idx_gate}, {qr, kv, ik, ig}, {kQr, kR, kID, kID}, in_x);
        fork();
        norm_cache_write<<<T, 256, 0, c->side>>>(kv, (const float *)w.kv_a_norm.data, g.rms_eps, ik,
                                                 (const float *)w.idx_k_norm_w.data,
                                                 (const float *)w.idx_k_norm_b.data, g.ln_eps, ig, dpos, pos_off,
                                                 c->cap, c->latent, c->kg);
        write_pools(c->side);
        rmsnorm_rows_q8<kQr><<<T, 256, 0, s>>>(qr, (const float *)w.q_a_norm.data, g.rms_eps, mm_in2);
        const MatmulInput in_qr = matmul_input_q8(qr, kQr, T, mm_in2);
        fork();
        run_on(c->side, w.idx_q_b, in_qr, iq, kIH * kID, 0);
        score_select(c->side);
        MLA_CUDA(cudaEventRecord(c->ev_join, c->side));
        run(w.q_b, in_qr, q, kH * kDq, 0);
        MLA_CUDA(cudaStreamWaitEvent(s, c->ev_iw, 0));   // idx_proj read in_x in mm_in, which Wk_b's prepare reuses
        q_abs(s);
    } else {
        run_group({&w.q_a, &w.kv_a, &w.idx_k, &w.idx_gate, &w.idx_proj}, {qr, kv, ik, ig, iw},
                  {kQr, kR, kID, kID, kIH}, in_x);
        scale_rows<<<(T * kIH + 255) / 256, 256, 0, s>>>(iw, (int64_t)T * kIH, iw_scale);
        rmsnorm_rows<<<T, 256, 0, s>>>(qr, kQr, (const float *)w.q_a_norm.data, g.rms_eps);
        norm_kv(s);
        run_group({&w.q_b, &w.idx_q_b}, {q, iq}, {kH * kDq, kIH * kID},
                  matmul_prepare(qr, kQr, T, 1, kQr, 0, mm_in, s));
        if (fast) {
            // Wk_b on the side stream, concurrently with the cache writes and the indexer scoring / selection;
            // the matmul scratch is idle on the main stream until Wv_b, and nothing there goes through cuBLAS
            fork();
            q_abs(c->side);
            MLA_CUDA(cudaEventRecord(c->ev_join, c->side));
            write_caches(s);
            score_select(s);
        } else {
            q_abs(s);
            write_caches(s);
        }
    }

    if (!fast) {
        if (!shared) {
            if (nv_max > 0) {
                dim3 grid((unsigned)((nv_max + kScorePoolsPerBlock - 1) / kScorePoolsPerBlock), T);
                score_pools<<<grid, kScoreWarps * 32, 0, s>>>(iq, iw, c->pooled, pos_off, L.n_pool_ld, score);
            }
            select_cells<<<T, kSelThreads, 0, s>>>(score, L.n_pool_ld, pos_off, top, sel_own, nsel_own);
        }
        attn_sparse<<<dim3(T, kH / kHG), kAttnThreads, 0, s>>>(qabs, c->latent, sel, nsel, sel_ld,
                                                                1.f / std::sqrt((float)kDq), o);
    } else {
        MLA_CUDA(cudaStreamWaitEvent(s, c->ev_join, 0));
        if (T <= kSplitMaxT) {
            // split over cells (one wave of blocks at 128K), partials merged by attn_combine
            const int split = T == 1 ? kFlashSplit1 : kFlashSplitN;
            const int n_split = (kSelMax + split - 1) / split;
            attn_flash<<<dim3(n_split, kH / kPfHeads, T), 256, kPfSmem, s>>>(qh, c->latent, sel, nsel, sel_ld, split,
                                                                            n_split, o, part_o, part_ml);
            attn_combine<<<dim3(kH, T), kCbThreads, 0, s>>>(part_o, part_ml, nsel, split, n_split, o,
                                                            fused ? mm_in2 : nullptr);
        } else {
            attn_flash<<<dim3(1, kH / kPfHeads, T), 256, kPfSmem, s>>>(qh, c->latent, sel, nsel, sel_ld, kSelMax, 1, o,
                                                                      nullptr, nullptr);
        }
    }

    // per-head Wv_b [512 -> 256] into the dead q buffer, then attn_output
    const MatmulInput in_o = fused && T <= kSplitMaxT ? matmul_input_q8(o, kR, T, mm_in2, kH, kH * kR, kR)
                                                      : matmul_prepare(o, kR, T, kH, kH * kR, kR, mm_in, s);
    run(w.v_b, in_o, q, kH * kDq, kDq);
    run(w.out, matmul_prepare(q, kH * kDq, T, 1, kH * kDq, 0, mm_in, s), out, D, 0);
    MLA_CUDA(cudaGetLastError());
}

}  // namespace

namespace {

void check_xin(const MatmulInput * xin, int T, int D) {
    if (xin && (xin->T != T || xin->n_in != D || xin->batch != 1)) {
        std::fprintf(stderr, "mla: prepared x input is [%d x %d] batch %d, expected [%d x %d] batch 1\n", xin->T, xin->n_in,
                     xin->batch, T, D);
        std::abort();
    }
}

void check_shared(const MlaSelection * shared) {
    if (shared && (!shared->sel || !shared->nsel || shared->sel_ld < 1)) {
        std::fprintf(stderr, "mla: shared selection needs sel, nsel and sel_ld >= 1\n");
        std::abort();
    }
}

MlaSelection chunk_selection(const MlaSelection & sh, int t0) {
    MlaSelection r = sh;
    r.sel += (int64_t)t0 * sh.sel_ld;
    r.nsel += t0;
    return r;
}

}  // namespace

void mla_forward(const Geometry & g, const MlaWeights & w, MlaCache * c, const Batch & b, const float * x_norm,
                 float * out, void * scratch, cudaStream_t s, const MlaSelection * shared, const MatmulInput * xin) {
    check_shared(shared);
    check_xin(xin, b.T, g.n_embd);
    if (xin && b.T > kChunk) {
        std::fprintf(stderr, "mla: a prepared x input needs T <= %d (one sub-chunk)\n", kChunk);
        std::abort();
    }
    check_geometry(g);
    if (b.pos0 < 0 || b.pos0 > c->n || b.pos0 + b.T > c->cap) {
        std::fprintf(stderr, "mla: batch [%lld, +%d) does not fit cache n=%lld cap=%lld\n", (long long)b.pos0, b.T,
                     (long long)c->n, (long long)c->cap);
        std::abort();
    }
    set_kernel_attributes();
    const int D = g.n_embd;
    const int TC = std::min(std::max(b.T, 1), kChunk);
    const Layout L = layout(TC, c->cap, D);
    kg_cut(c, b.pos0);
    c->n = b.pos0;
    for (int t0 = 0; t0 < b.T; t0 += TC) {
        const int T = std::min(TC, b.T - t0);
        const int64_t pos0 = b.pos0 + t0;
        const MlaSelection sh = shared ? chunk_selection(*shared, t0) : MlaSelection{};
        forward_chunk(g, w, c, L, T, nullptr, pos0, x_norm + (int64_t)t0 * D, out + (int64_t)t0 * D,
                      static_cast<char *>(scratch), s, shared ? &sh : nullptr, xin);
        c->n = pos0 + T;
        c->last_T = T;
        c->last_TC = TC;
        c->last_pos0 = pos0;
    }
}

void mla_forward_dev(const Geometry & g, const MlaWeights & w, MlaCache * c, int T, const int64_t * d_pos0,
                     const float * x_norm, float * out, void * scratch, cudaStream_t s, const MlaSelection * shared,
                     const MatmulInput * xin) {
    check_geometry(g);
    check_shared(shared);
    check_xin(xin, T, g.n_embd);
    if (T < 1 || T > kChunk || T > c->cap || !d_pos0) {
        std::fprintf(stderr, "mla: mla_forward_dev needs 1 <= T <= %d (got %d), T <= capacity and a device position\n",
                     kChunk, T);
        std::abort();
    }
    set_kernel_attributes();
    const Layout L = layout(T, c->cap, g.n_embd);
    forward_chunk(g, w, c, L, T, d_pos0, 0, x_norm, out, static_cast<char *>(scratch), s, shared, xin);
    c->last_T = T;
    c->last_TC = T;
    c->last_pos0 = -1;   // known on device only; mla_cache_set_size fills it in
}

void mla_cache_set_size(MlaCache * c, int64_t n_tokens) {
    if (n_tokens < 0 || n_tokens > c->cap) {
        std::fprintf(stderr, "mla: set_size %lld outside [0, %lld]\n", (long long)n_tokens, (long long)c->cap);
        std::abort();
    }
    if (n_tokens < c->n) kg_cut(c, n_tokens);
    c->n = n_tokens;
    if (c->last_pos0 < 0) c->last_pos0 = n_tokens - c->last_T;
}

// ---- prefix blobs ----------------------------------------------------------------------------------------

namespace {
constexpr uint32_t kBlobMagic = 0x424c4d47u;   // "GMLB"
enum : uint32_t { kBlobRows = 1, kBlobTail = 2 };
struct BlobHeader {
    uint32_t magic, kind;
    int32_t  latent, pad;
    int64_t  a, b;
};
constexpr size_t kBlobHeader = 64;   // keeps the row data 16-byte aligned
static_assert(sizeof(BlobHeader) <= kBlobHeader, "blob header");
constexpr size_t kKgRowBytes = sizeof(__half) * 2 * kID;

void put_header(void * host, uint32_t kind, MlaLatent t, int64_t a, int64_t b) {
    BlobHeader h{kBlobMagic, kind, (int32_t)t, 0, a, b};
    std::memset(host, 0, kBlobHeader);
    std::memcpy(host, &h, sizeof h);
}
bool header_is(const void * host, uint32_t kind, MlaLatent t, int64_t a, int64_t b) {
    BlobHeader h;
    std::memcpy(&h, host, sizeof h);
    return h.magic == kBlobMagic && h.kind == kind && h.latent == (int32_t)t && h.a == a && h.b == b;
}
void check_range(const MlaCache * c, int64_t a, int64_t b, int64_t hi, const char * what) {
    if (a < 0 || a > b || b > hi) {
        std::fprintf(stderr, "mla: %s [%lld, %lld) outside [0, %lld]\n", what, (long long)a, (long long)b,
                     (long long)hi);
        std::abort();
    }
    (void)c;
}
// the import replaced rows: drop the ring (another sequence's rows may sit at these positions) and the taps
void after_import(MlaCache * c, cudaStream_t s) {
    MLA_CUDA(cudaMemsetAsync(c->kg.tag, 0xff, sizeof(int64_t) * kKgRing, s));
    c->last_T = 0;
    c->last_TC = 0;
    c->last_pos0 = 0;
}
}  // namespace

size_t mla_cache_rows_bytes(const MlaCache * c, int64_t a, int64_t b) {
    check_range(c, a, b, c->cap, "rows blob");
    return kBlobHeader + latent_row_bytes(c->latent_type) * (size_t)(b - a) +
           sizeof(__half) * kID * (size_t)(b / kPool - a / kPool);
}

void mla_cache_export_rows(const MlaCache * c, int64_t a, int64_t b, void * host, cudaStream_t s) {
    check_range(c, a, b, c->n, "export rows");
    put_header(host, kBlobRows, c->latent_type, a, b);
    char * p = static_cast<char *>(host) + kBlobHeader;
    const size_t n = (size_t)(b - a), np = (size_t)(b / kPool - a / kPool);
    if (c->latent.q8) {
        MLA_CUDA(cudaMemcpyAsync(p, c->latent.q8 + a * kR, n * kR, cudaMemcpyDeviceToHost, s));
        p += n * kR;
        const size_t sb = sizeof(__half) * (kR / kQ8Group);
        MLA_CUDA(cudaMemcpyAsync(p, c->latent.scale + a * (kR / kQ8Group), n * sb, cudaMemcpyDeviceToHost, s));
        p += n * sb;
    } else {
        MLA_CUDA(cudaMemcpyAsync(p, c->latent.f16 + a * kR, n * kR * sizeof(__half), cudaMemcpyDeviceToHost, s));
        p += n * kR * sizeof(__half);
    }
    if (np > 0)
        MLA_CUDA(cudaMemcpyAsync(p, c->pooled + (a / kPool) * kID, np * kID * sizeof(__half), cudaMemcpyDeviceToHost,
                                 s));
}

bool mla_cache_import_rows(MlaCache * c, int64_t a, int64_t b, const void * host, cudaStream_t s) {
    check_range(c, a, b, c->cap, "import rows");
    if (a > c->n) {
        std::fprintf(stderr, "mla: import rows [%lld, %lld) past the cache size %lld\n", (long long)a, (long long)b,
                     (long long)c->n);
        std::abort();
    }
    if (!header_is(host, kBlobRows, c->latent_type, a, b)) return false;
    const char * p = static_cast<const char *>(host) + kBlobHeader;
    const size_t n = (size_t)(b - a), np = (size_t)(b / kPool - a / kPool);
    if (c->latent.q8) {
        MLA_CUDA(cudaMemcpyAsync(c->latent.q8 + a * kR, p, n * kR, cudaMemcpyHostToDevice, s));
        p += n * kR;
        const size_t sb = sizeof(__half) * (kR / kQ8Group);
        MLA_CUDA(cudaMemcpyAsync(c->latent.scale + a * (kR / kQ8Group), p, n * sb, cudaMemcpyHostToDevice, s));
        p += n * sb;
    } else {
        MLA_CUDA(cudaMemcpyAsync(c->latent.f16 + a * kR, p, n * kR * sizeof(__half), cudaMemcpyHostToDevice, s));
        p += n * kR * sizeof(__half);
    }
    if (np > 0)
        MLA_CUDA(cudaMemcpyAsync(c->pooled + (a / kPool) * kID, p, np * kID * sizeof(__half), cudaMemcpyHostToDevice,
                                 s));
    // rows >= a are replaced: the hole keeps its part below a and gains [a, b) (the tail import trims its end)
    c->kg_hole_b = std::min(c->kg_hole_b, a);
    c->kg_hole_a = c->kg_hole_b > c->kg_hole_a ? c->kg_hole_a : a;
    c->kg_hole_b = b;
    if (c->kg_hole_b <= c->kg_hole_a) c->kg_hole_a = c->kg_hole_b = 0;
    c->n = b;
    after_import(c, s);
    return true;
}

size_t mla_cache_tail_bytes(const MlaCache *) { return kBlobHeader + (kPool - 1) * kKgRowBytes; }

void mla_cache_export_tail(const MlaCache * c, int64_t n, void * host, cudaStream_t s) {
    if (n < 0 || n > c->n || !kg_tail_held(c, n)) {
        std::fprintf(stderr, "mla: export tail at %lld (size %lld, key | gate rows held: %d)\n", (long long)n,
                     (long long)c->n, (int)kg_tail_held(c, n));
        std::abort();
    }
    const int64_t lo = n / kPool * kPool;
    put_header(host, kBlobTail, c->latent_type, lo, n);
    if (n > lo)
        MLA_CUDA(cudaMemcpyAsync(static_cast<char *>(host) + kBlobHeader, c->kg_host + lo * 2 * kID,
                                 (size_t)(n - lo) * kKgRowBytes, cudaMemcpyHostToHost, s));
}

bool mla_cache_import_tail(MlaCache * c, int64_t n, const void * host, cudaStream_t s) {
    if (n != c->n) {
        std::fprintf(stderr, "mla: import tail at %lld, cache size %lld\n", (long long)n, (long long)c->n);
        std::abort();
    }
    const int64_t lo = n / kPool * kPool;
    if (!header_is(host, kBlobTail, c->latent_type, lo, n)) return false;
    if (n > lo)
        MLA_CUDA(cudaMemcpyAsync(c->kg_host + lo * 2 * kID, static_cast<const char *>(host) + kBlobHeader,
                                 (size_t)(n - lo) * kKgRowBytes, cudaMemcpyHostToHost, s));
    c->kg_hole_b = std::min(c->kg_hole_b, lo);
    if (c->kg_hole_b <= c->kg_hole_a) c->kg_hole_a = c->kg_hole_b = 0;
    after_import(c, s);
    return true;
}

}  // namespace strata::glm
